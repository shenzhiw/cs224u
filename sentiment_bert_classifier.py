# Import modules
import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, AutoTokenizer
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
from sklearn.metrics import classification_report

def get_batch_token_ids(batch, tokenizer):
    """Map `batch` to a tensor of ids. The return
    value should meet the following specification:

    1. The max length should be 512.
    2. Examples longer than the max length should be truncated
    3. Examples should be padded to the max length for the batch.
    4. The special [CLS] should be added to the start and the special
       token [SEP] should be added to the end.
    5. The attention mask should be returned
    6. The return value of each component should be a tensor.

    Parameters
    ----------
    batch: list of str
    tokenizer: Hugging Face tokenizer

    Returns
    -------
    dict with at least "input_ids" and "attention_mask" as keys,
    each with Tensor values

    """
    batch_encoding = tokenizer.batch_encode_plus(
        batch,
        max_length=512,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors='pt'
        )
    return batch_encoding

class BertClassifierModule(nn.Module):
    """
    A BERT-based classifier module for sentiment analysis.
    Args:
        n_classes (int): Number of output classes for classification. Default is 3.
        hidden_activation (nn.Module): Activation function to use in the hidden layer. Default is nn.ReLU().
        weights_name (str): Name of the pre-trained BERT model to use. Default is 'prajjwal1/bert-mini'.
        dropout_rate (float): Dropout rate to use in the dropout layer. Default is 0.2.
    Attributes:
        n_classes (int): Number of output classes for classification.
        weights_name (str): Name of the pre-trained BERT model to use.
        bert (AutoModel): Pre-trained BERT model.
        dropout (nn.Dropout): Dropout layer.
        hidden_activation (nn.Module): Activation function to use in the hidden layer.
        hidden_dim (int): Dimensionality of the hidden layer.
        classifier_layer (nn.Sequential): Sequential container for the classifier layers.
    Methods:
        forward(indices, mask):
            Performs a forward pass of the classifier.
            Args:
                indices (torch.Tensor): Input tensor containing token indices.
                mask (torch.Tensor): Attention mask tensor.
            Returns:
                torch.Tensor: Output tensor containing class logits.
    """
    def __init__(self,
            n_classes=3,
            hidden_activation=nn.ReLU(),
            weights_name='prajjwal1/bert-mini',
            dropout_rate=0.2):
        super().__init__()
        self.n_classes = n_classes
        self.weights_name = weights_name
        self.bert = AutoModel.from_pretrained(self.weights_name, output_hidden_states=True)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer
        self.bert.train()
        self.hidden_activation = hidden_activation
        self.hidden_dim = self.bert.embeddings.word_embeddings.embedding_dim
        self.classifier_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            self.hidden_activation,
            nn.Linear(self.hidden_dim, self.n_classes)
        )
    def forward(self, indices, mask):
        reps = self.bert(indices, attention_mask=mask)
        # use the second-to-last hidden layer
        hidden_states = reps.hidden_states[-2]
        mask_resize = mask.unsqueeze(-1).expand(hidden_states.size()).float()
        # mean pooling
        mean_pooled = torch.sum(hidden_states * mask_resize, dim=1) / torch.sum(mask_resize, dim=1)
        # max pooling
        hidden_states[mask_resize == 0] = -float('inf')  # set padding tokens to -inf
        max_pooled = torch.max(hidden_states, dim=1)[0]
        # mean + max pooling, plus dropout
        pooled_output = self.dropout(torch.cat([mean_pooled, max_pooled], dim=1))
        return self.classifier_layer(pooled_output)

class BertClassifier(TorchShallowNeuralClassifier):
    """
    A classifier that uses a pre-trained BERT model for sentiment analysis.
    Parameters
    ----------
    weights_name : str
        The name of the pre-trained BERT model weights to use.
    *args : tuple
        Additional positional arguments to pass to the parent class.
    **kwargs : dict
        Additional keyword arguments to pass to the parent class.
    Methods
    -------
    build_graph()
        Constructs the neural network graph using the BERT model.
    build_dataset(X, y=None)
        Converts input data into a format suitable for the BERT model.
        If `y` is provided, it also processes the labels.
    """

    def __init__(self, weights_name, *args, **kwargs):
        self.weights_name = weights_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.weights_name)
        super().__init__(*args, **kwargs)
        self.params += ['weights_name']

    def build_graph(self):
        return BertClassifierModule(
            self.n_classes_, self.hidden_activation, self.weights_name)

    def build_dataset(self, X, y=None):
        data = get_batch_token_ids(X, self.tokenizer)
        if y is None:
            dataset = torch.utils.data.TensorDataset(
                data['input_ids'], data['attention_mask'])
        else:
            self.classes_ = sorted(set(y))
            self.n_classes_ = len(self.classes_)
            class2index = dict(zip(self.classes_, range(self.n_classes_)))
            y = [class2index[label] for label in y]
            y = torch.tensor(y)
            dataset = torch.utils.data.TensorDataset(
                data['input_ids'], data['attention_mask'], y)
        return dataset
