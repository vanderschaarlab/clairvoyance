"""Attention-based time-series module.

Time-series prediction with attention on top of RNN.
Only applicable to one-shot prediction.
"""

# Necessary packages
import os
import tensorflow as tf
from datetime import datetime
from keras.callbacks import ModelCheckpoint
from utils import binary_cross_entropy_loss, mse_loss, rnn_layer
from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate
from tensorflow.keras import Input, Model
from base import BaseEstimator, PredictorMixin


class Attention(BaseEstimator, PredictorMixin):
    """Attention model on top of RNN networks for time-series prediction.
    
    Attributes:
        - task: classification or regression
        - model_type: 'rnn', 'lstm', or 'gru'
        - h_dim: hidden dimensions
        - n_layer: the number of layers
        - batch_size: the number of samples in each batch
        - epoch: the number of iteration epochs
        - static_mode: 'concatenate' or None
        - time_mode: 'concatenate' or None
        - model_id: the name of model
        - model_path: model path for saving
        - verbose: print intermediate process
    """

    def __init__(
        self,
        task=None,
        model_type=None,
        h_dim=None,
        n_layer=None,
        batch_size=None,
        epoch=None,
        learning_rate=None,
        static_mode=None,
        time_mode=None,
        model_id="attention_model",
        model_path="tmp",
        verbose=False,
    ):

        super().__init__(task)

        self.task = task
        self.model_type = model_type

        self.h_dim = h_dim
        self.n_layer = n_layer
        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.static_mode = static_mode
        self.time_mode = time_mode
        self.model_path = model_path
        self.model_id = model_id
        self.verbose = verbose

        # Predictor model & optimizer define
        self.attention_model = None
        self.adam = None

        # Set path for model saving
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.save_file_name = "{}/{}".format(model_path, model_id) + datetime.now().strftime("%H%M%S") + ".hdf5"

    def new(self, model_id):
        return Attention(
            self.task,
            self.model_type,
            self.h_dim,
            self.n_layer,
            self.batch_size,
            self.epoch,
            self.learning_rate,
            self.static_mode,
            self.time_mode,
            model_id,
            self.model_path,
            self.verbose,
        )

    @staticmethod
    def get_hyperparameter_space():
        hyp_ = [
            {"name": "h_dim", "type": "discrete", "domain": list(range(10, 150, 10)), "dimensionality": 1},
            {"name": "n_layer", "type": "discrete", "domain": list(range(1, 4, 1)), "dimensionality": 1},
            {"name": "learning_rate", "type": "continuous", "domain": [0.0005, 0.01], "dimensionality": 1},
        ]

        return hyp_

    def attention_3d_block(self, hidden_states):
        """Attention mechanism.
        
        Reference - https://github.com/philipperemy/keras-attention-mechanism
        
        Args:
            - hidden_states: RNN hidden states (3d array)
            
        Return:
            - attention_vector: output states after attention mechanism.
        """
        # hidden_states.shape = (batch_size, time_steps, hidden_size)
        hidden_size = int(hidden_states.shape[2])
        # Inside dense layer
        #                  hidden_states                dot                   W                =>               score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = Dense(hidden_size, use_bias=False, name="attention_score_vec")(hidden_states)
        #                        score_first_part                     dot                last_hidden_state         => attention_weights
        # (batch_size, time_steps, hidden_size) dot     (batch_size, hidden_size)    => (batch_size, time_steps)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name="last_hidden_state")(hidden_states)
        score = dot([score_first_part, h_t], [2, 1], name="attention_score")
        attention_weights = Activation("softmax", name="attention_weight")(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = dot([hidden_states, attention_weights], [1, 1], name="context_vector")
        pre_activation = concatenate([context_vector, h_t], name="attention_output")
        attention_vector = Dense(self.h_dim, use_bias=False, activation="tanh", name="attention_vector")(pre_activation)
        return attention_vector

    def _build_model(self, x, y):
        """Construct the model using feature and label statistics.
        
        Args:
            - x: temporal feature
            - y: labels
            
        Returns:
            - model: predictor model
        """
        self.model_type = "gru"

        # Only for one-shot prediction
        assert len(y.shape) == 2

        # Parameters
        dim = len(x[0, 0, :])
        seq_len = len(x[0, :, 0])

        self.adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

        inputs = Input(shape=(seq_len, dim,))
        rnn_out = rnn_layer(inputs, self.model_type, self.h_dim, return_seq=True)
        for _ in range(self.n_layer - 1):
            rnn_out = rnn_layer(rnn_out, self.model_type, self.h_dim, return_seq=True)

        attention_output = self.attention_3d_block(rnn_out)

        if self.task == "classification":
            output = Dense(y.shape[-1], activation="sigmoid", name="output")(attention_output)
            attention_model = Model(inputs=[inputs], outputs=[output])
            attention_model.compile(loss=binary_cross_entropy_loss, optimizer=self.adam)
        elif self.task == "regression":
            output = Dense(y.shape[-1], activation="linear", name="output")(attention_output)
            attention_model = Model(inputs=[inputs], outputs=[output])
            attention_model.compile(loss=mse_loss, optimizer=self.adam, metrics=["mse"])

        return attention_model

    def fit(self, dataset, fold=0, train_split="train", valid_split="val"):
        """Fit the attention model.
        
        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Cross validation fold
            - train_split: training set splitting parameter
            - valid_split: validation set splitting parameter
            
        Returns:
            - self.attention_model: trained attention model
        """
        # Train/Valid datasets
        train_x, train_y = self._data_preprocess(dataset, fold, train_split)
        valid_x, valid_y = self._data_preprocess(dataset, fold, valid_split)

        # Build attention model
        self.attention_model = self._build_model(train_x, train_y)

        # Callback for the best model saving
        save_best = ModelCheckpoint(
            self.save_file_name, monitor="val_loss", mode="min", verbose=False, save_best_only=True
        )

        # Train the model
        self.attention_model.fit(
            train_x,
            train_y,
            batch_size=self.batch_size,
            epochs=self.epoch,
            validation_data=(valid_x, valid_y),
            callbacks=[save_best],
            verbose=self.verbose,
        )

        self.attention_model.load_weights(self.save_file_name)
        os.remove(self.save_file_name)

        return self.attention_model

    def predict(self, dataset, fold=0, test_split="test"):
        """Predict on the new dataset by the trained model.
        
        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Cross validation fold
            - test_split: testing set splitting parameter
            
        Returns:
            - test_y_hat: predictions on the new dataset
        """
        test_x, _ = self._data_preprocess(dataset, fold, test_split)
        test_y_hat = self.attention_model.predict(test_x)
        return test_y_hat

    def save_model(self, model_path):
        """Save the model to model_path
        
        Args:
            - model_path: path of the saved model (it should be .h5)
        """
        assert model_path[-3:] == ".h5"
        self.attention_model.save(model_path)

    def load_model(self, model_path):
        """Load and return the model from model_path        
        
        Args:
            - model_path: path of the saved model (it should be .h5)
        """
        assert model_path[-3:] == ".h5"
        assert os.path.exists(model_path) is True

        loaded_model = tf.keras.models.load_model(model_path, compile=False)

        if self.task == "classification":
            loaded_model.compile(loss=binary_cross_entropy_loss, optimizer=self.adam)
        elif self.task == "regression":
            loaded_model.compile(loss=mse_loss, optimizer=self.adam)

        self.attention_model = loaded_model

        return loaded_model
