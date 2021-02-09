"""Utility functions for models.

Loss functions:
    (1) binary_cross_entropy_loss
    (2) mse_loss
    (3) rmse_loss
    (4) select_loss    
    
Architectures:
    (1) rnn_layer
    (2) rnn_sequential
    
Others:
    (1) compose: compose multiple functions
    (2) PipelineComposer: compose multiple functions in pipeline
"""

# Necessary packages
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU


def binary_cross_entropy_loss(y_true, y_pred):
    """User defined cross entropy loss.
    
    Args:
        - y_true: true labels
        - y_pred: predictions
        
    Returns:
        - loss: computed loss
    """
    # Exclude masked labels
    idx = tf.cast((y_true >= 0), float)
    # Cross entropy loss excluding masked labels
    loss = -(idx * y_true * tf.math.log(y_pred + 1e-6) + idx * (1 - y_true) * tf.math.log(1 - y_pred + 1e-6))
    return loss


def mse_loss(y_true, y_pred):
    """User defined mean squared loss.
    
    Args:
        - y_true: true labels
        - y_pred: predictions
        
    Returns:
        - loss: computed loss
    """
    # Exclude masked labels
    idx = tf.cast((y_true >= 0), float)
    # Mean squared loss excluding masked labels
    loss = idx * ((y_true - y_pred) ** 2)
    return loss


def select_loss(y_true, y_pred):
    """User defined selection loss.
    
    Args:
        - y_true: true labels
        - y_pred: predictions
        
    Returns:
        - loss: computed selection loss
    """
    # Exclude masked labels
    idx = tf.cast((y_true >= 0), float)
    # The average value of selected important samples
    loss = idx * y_pred
    return loss


def rmse_loss(y_true, y_pred):
    """User defined root mean squared loss.
    
    Args:
        - y_true: true labels
        - y_pred: predictions
        
    Returns:
        - loss: computed loss
    """
    # Exclude masked labels
    idx = tf.cast((y_true >= 0), float)
    # Root mean squared loss excluding masked labels
    loss = tf.sqrt(idx * tf.pow(y_true - y_pred, 2) + 1e-8)

    return loss


def rnn_layer(input_layer, model_name, h_dim, return_seq):
    """Add one rnn layer.
    
    Args:
        - input_layer
        - model_name: rnn, lstm, or gru
        - h_dim: hidden state dimensions
        - return_seq: True or False
        
    Returns:
        - output_layer
    """
    if model_name == "rnn":
        output_layer = SimpleRNN(h_dim, return_sequences=return_seq)(input_layer)
    elif model_name == "lstm":
        output_layer = LSTM(h_dim, return_sequences=return_seq)(input_layer)
    elif model_name == "gru":
        output_layer = GRU(h_dim, return_sequences=return_seq)(input_layer)

    return output_layer


def rnn_sequential(model, model_name, h_dim, return_seq, name=None):
    """Add one rnn layer in sequential model.
    
    Args:
        - model: sequential rnn model
        - model_name: rnn, lstm, or gru
        - h_dim: hidden state dimensions
        - return_seq: True or False
        - name: layer name
        
    Returns:
        - model: sequential rnn model
    """
    if name == None:
        if model_name == "rnn":
            model.add(layers.SimpleRNN(h_dim, return_sequences=return_seq))
        elif model_name == "lstm":
            model.add(layers.LSTM(h_dim, return_sequences=return_seq))
        elif model_name == "gru":
            model.add(layers.GRU(h_dim, return_sequences=return_seq))
    else:
        if model_name == "rnn":
            model.add(layers.SimpleRNN(h_dim, return_sequences=return_seq, name=name))
        elif model_name == "lstm":
            model.add(layers.LSTM(h_dim, return_sequences=return_seq, name=name))
        elif model_name == "gru":
            model.add(layers.GRU(h_dim, return_sequences=return_seq, name=name))

    return model


def compose(*functions):
    """Compose multiple functions.
    
    Args:
        - functions: functions for composing
        
    Returns:
        - inner: composed function
    """

    def inner(arg):
        for f in functions:
            arg = f(arg)
        return arg

    return inner


class PipelineComposer:
    """Composing a pipeline from stages.

        Attributes:
            - *stage: individual stages in the pipeline
        """

    def __init__(self, *stage):
        self.stage = stage

    def fit(self, dataset):
        """Fit the whole pipeline.

        Args:
            - dataset: Input data for fitting
        """
        for s in self.stage:
            s.fit(dataset)

    def transform(self, dataset):
        """Use the whole pipeline to transform the data set.

        Args:
            - dataset: Input data for transform
        """
        for s in self.stage:
            dataset = s.transform(dataset)
        return dataset

    def fit_transform(self, dataset):
        """Fit the whole pipeline and apply the transform.

        Args:
            - dataset: Input data for fit and transform
        """
        for s in self.stage:
            dataset = s.fit_transform(dataset)
        return dataset
