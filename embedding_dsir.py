from typing import List, Optional, Dict, Callable, Union, Iterable
import hashlib
from tqdm import tqdm
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize
from nltk import ngrams as get_ngrams
from sentence_transformers import SentenceTransformer
import numpy as np

from data_selection.base import (
        DSIR,
        default_load_dataset_fn,
        default_parse_example_fn,
        _iterate_virtually_sharded_dataset,
)

from data_selection.utils import parallelize


wpt = WordPunctTokenizer()


def hash_buckets(text: str, num_buckets: int = 10000) -> int:
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % num_buckets


def get_ngram_counts(line: str,
                     n: int = 2,
                     num_buckets: int = 10000,
                     counts: Optional[np.ndarray] = None,
                     tokenizer: Callable = wpt.tokenize) -> np.ndarray:
    '''Return ngram count features given a string.

    Args:
        line: string to get ngram counts from
        n: n in ngrams
        num_buckets: number of buckets to hash ngrams into
        counts: pre-initialized counts array
        tokenizer: tokenization function to use. Defaults to word_tokenize from nltk
    '''
    words = tokenizer(line.lower())

    if counts is None:
        counts = np.zeros(num_buckets, dtype=int)

    for w in words:
        counts[hash_buckets(w, num_buckets=num_buckets)] += 1
    for i in range(2, n + 1):
        for ng in list(get_ngrams(words, i)):
            ng = ' '.join(ng)
            counts[hash_buckets(ng, num_buckets=num_buckets)] += 1
    return counts

def get_setence_embedding(sentence, model = "all-MiniLM-L6-v2"):
    """
    Generate an embedding for a given sentence using a specified model.
    Args:
        sentence (str): The sentence to be embedded.
        model (str, optional): The name of the pre-trained model to use for generating the embedding. 
                               Defaults to "all-MiniLM-L6-v2".
    Returns:
        numpy.ndarray: The embedding of the input sentence.
    """
    sentence_model = SentenceTransformer(model)
    return sentence_model.encode(sentence)

class EmbeddingDSIR(DSIR):
    """DSIR with hashed n-gram features."""

    def __init__(self,
                 raw_datasets: List[str],
                 target_datasets: List[str],
                 cache_dir: str,
                 raw_load_dataset_fn: Callable[[str], Iterable[Dict]] = default_load_dataset_fn,
                 raw_parse_example_fn: Callable[[Dict], str] = default_parse_example_fn,
                 target_load_dataset_fn: Callable[[str], Iterable[Dict]] = default_load_dataset_fn,
                 target_parse_example_fn: Callable[[Dict], str] = default_parse_example_fn,
                 num_proc: Optional[int] = None,
                 ngrams: int = 2,
                 num_buckets: int = 384,
                 tokenizer: str = 'wordpunct',
                 min_example_length: int = 100,
                 target_laplace_smoothing: float = 0.0,
                 separate_targets: bool = False,
                 target_proportions: Optional[List[float]] = None) -> None:
        '''Initialize the HashedNgramDSIR object.

        Args:
            raw_datasets: List of data paths
            target_datasets: List of data paths
            cache_dir: place to store cached log_importance_weights
            load_dataset_fn: Function to load a dataset from a path. Defaults to default_load_dataset_fn.
            parse_example_fn: Function that takes in an example dict and returns a string.
                              Defaults to returning the 'text' field of the example.
            num_proc: number of processes to use for parallelization. Defaults to number of cores.
            ngrams: N in N-grams. 2 means both unigram and bigrams.
            num_buckets: number of buckets to hash ngrams into.
            tokenizer: word_tokenize or wordpunct
            min_example_length: minimum number of tokens in an example to be considered.
            target_laplace_smoothing: Smooth the target hashed ngram distribution. This parameter is a pseudo-count. This could be useful for small target datasets.
            separate_targets: whether to select data separately for each target and then join them
            target_proportions: weighting across multiple targets if separate_targets=True. Set to None to weight by the size of each target dataset
        '''
        super().__init__(
                raw_datasets=raw_datasets,
                target_datasets=target_datasets,
                cache_dir=cache_dir,
                raw_load_dataset_fn=raw_load_dataset_fn,
                raw_parse_example_fn=raw_parse_example_fn,
                target_load_dataset_fn=target_load_dataset_fn,
                target_parse_example_fn=target_parse_example_fn,
                num_proc=num_proc,
                separate_targets=separate_targets,
                target_proportions=target_proportions)
        if tokenizer == 'word_tokenize':
            self.tokenizer = word_tokenize
        elif tokenizer == 'wordpunct':
            self.tokenizer = wpt.tokenize
        else:
            raise ValueError('tokenizer not recognized')
        self.ngrams = ngrams
        self.num_buckets = num_buckets
        self.min_example_length = min_example_length
        self.raw_probs = None
        self.target_probs = None
        self.log_diff = None
        self.target_laplace_smoothing = target_laplace_smoothing

    def featurizer(self, text: str) -> np.ndarray:
        # return get_ngram_counts(text, tokenizer=self.tokenizer, num_buckets=self.num_buckets, n=self.ngrams)
        return get_setence_embedding(text)

    def importance_estimator(self, features: np.ndarray) -> Union[float, np.ndarray]:
        return np.dot(self.log_diff, features)

    # def get_perexample_metadata(self, ex: Dict, features: np.ndarray) -> int:
    #     """Returns the example length."""
    #     remainder = self.ngrams * (self.ngrams - 1) / 2
    #     return (features.sum() + remainder) // self.ngrams

    # def perexample_metadata_filter(self, concat_metadata: np.ndarray) -> np.array:
    #     """Filters out short examples."""
    #     return concat_metadata >= self.min_example_length

    def _fit_bow(self,
                 paths: List[str],
                 load_dataset_fn: Callable[[str], Iterable[Dict]] = default_load_dataset_fn,
                 parse_example_fn: Callable[[Dict], str] = default_parse_example_fn) -> np.ndarray:

        sharded_datasets = self._get_virtually_sharded_datasets(paths)

        def job(args: Dict):
            path = args['path']
            num_shards = args['num_shards']
            shard_idx = args['shard_idx']

            setence_embedding = np.zeros(self.num_buckets).astype(int)
            dataset = load_dataset_fn(path)
            iterator = _iterate_virtually_sharded_dataset(dataset, num_shards, shard_idx)
            for ex in tqdm(iterator, miniters=10000, maxinterval=1000000):
                if parse_example_fn is not None:
                    text = parse_example_fn(ex)
                else:
                    text = ex
                setence_embedding = get_setence_embedding(text)

            return setence_embedding

        all_setence_embedding = parallelize(job, sharded_datasets, self.num_proc)
        setence_embedding_mean = sum(all_setence_embedding) / len(all_setence_embedding)

        return setence_embedding_mean

    def fit_importance_estimator(self) -> None:
        '''Fit the importance estimator.
        Args:
        '''

        self.raw_probs = self._fit_bow(
                self.raw_datasets,
                parse_example_fn=self.raw_parse_example_fn,
                load_dataset_fn=self.raw_load_dataset_fn)
        self.raw_probs = self.raw_probs / self.raw_probs.sum()

        if self.separate_targets:
            target_probs = []
            target_proportions = []

            for target_dataset in self.target_datasets:
                curr_target_probs = self._fit_bow(
                        [target_dataset],
                        parse_example_fn=self.target_parse_example_fn,
                        load_dataset_fn=self.target_load_dataset_fn)
                target_proportions.append(curr_target_probs.sum())
                # smoothing
                curr_target_probs = curr_target_probs + self.target_laplace_smoothing
                curr_target_probs = curr_target_probs / curr_target_probs.sum()
                target_probs.append(curr_target_probs)
            target_proportions = np.asarray(target_proportions)
            if self.target_proportions is None:
                self.target_proportions = target_proportions / target_proportions.sum()

            self.target_probs = np.asarray(target_probs)

        else:
            self.target_probs = self._fit_bow(
                    self.target_datasets,
                    parse_example_fn=self.target_parse_example_fn,
                    load_dataset_fn=self.target_load_dataset_fn)
            # smoothing
            self.target_probs = self.target_probs + self.target_laplace_smoothing
            self.target_probs = self.target_probs / self.target_probs.sum()


        self.log_diff = np.log(self.target_probs + 1e-8) - np.log(self.raw_probs + 1e-8)