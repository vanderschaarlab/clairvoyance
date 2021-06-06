"""Temporal CNN blocks.

Time-series prediction with temporal CNN.
Reference: https://github.com/philipperemy/keras-tcn
"""

# Necessary packages
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import binary_cross_entropy_loss, mse_loss
from tensorflow.keras.layers import Dense, TimeDistributed
from tensorflow.keras import Input, Model
from base import BaseEstimator, PredictorMixin

# Need download this package
from tcn import TCN


class TemporalCNN(BaseEstimator, PredictorMixin):
    """Temporal CNN model for for time-series prediction.
    
    Attributes:
        - task: classification or regression
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
        h_dim=None,
        n_layer=None,
        batch_size=None,
        epoch=None,
        learning_rate=None,
        static_mode=None,
        time_mode=None,
        model_id="tcn_model",
        model_path="tmp",
        verbose=False,
    ):

        super().__init__(task)

        self.task = task
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

        # Predictor model define
        self.tcn_model = None
        self.adam = None

        # Set path for model saving
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.save_file_name = "{}/{}".format(model_path, model_id) + datetime.now().strftime("%H%M%S") + ".hdf5"

    def new(self, model_id):
        """Create a new model with the same parameter as the existing one.
        
        Args:
            - model_id: an unique identifier for the new model
        Returns:
            - a new TemporalCNN
        """
        return TemporalCNN(
            self.task,
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
            {"name": "batch_size", "type": "discrete", "domain": list(range(100, 1001, 100)), "dimensionality": 1},
            {"name": "learning_rate", "type": "continuous", "domain": [0.0005, 0.01], "dimensionality": 1},
        ]

        return hyp_

    def _build_model(self, x, y):
        """Construct the model using feature and label statistics.
        
        Args:
            - x: temporal feature
            - y: labels
            
        Returns:
            - model: predictor model
        """
        # Parameters
        dim = len(x[0, 0, :])
        seq_len = len(x[0, :, 0])
        dim_y = len(y.shape)
        dilations = [2 ** (i) for i in range(int(np.log2(seq_len / 4)))]
        # Small hidden dimensions are better
        if self.h_dim > 30:
            self.h_dim = int(self.h_dim / 5)

        # Optimizer
        self.adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
        # Input
        inputs = Input(shape=(seq_len, dim,))
        # First layer
        tcn_out = TCN(nb_filters=self.h_dim, dilations=dilations, return_sequences=True, use_skip_connections=False)(inputs)

        # Multi-layer
        for _ in range(self.n_layer - 2):
            tcn_out = TCN(nb_filters=self.h_dim, dilations=dilations, return_sequences=True, use_skip_connections=False)(tcn_out)

        # For classification
        if self.task == "classification":
            # For online prediction
            if dim_y == 3:
                tcn_out = TCN(nb_filters=self.h_dim, dilations=dilations, return_sequences=True, use_skip_connections=False)(tcn_out)
                output = TimeDistributed(Dense(y.shape[-1], activation="sigmoid", name="output"))(tcn_out)
            # For one-shot prediction
            elif dim_y == 2:
                tcn_out = TCN(nb_filters=self.h_dim, dilations=dilations, return_sequences=False, use_skip_connections=False)(tcn_out)
                output = Dense(y.shape[-1], activation="sigmoid", name="output")(tcn_out)
            # Model define and compile
            tcn_model = Model(inputs=[inputs], outputs=[output])
            tcn_model.compile(loss=binary_cross_entropy_loss, optimizer=self.adam)
        # For regression
        elif self.task == "regression":
            # For online prediction
            if dim_y == 3:
                tcn_out = TCN(nb_filters=self.h_dim, dilations=dilations, return_sequences=True, use_skip_connections=False)(tcn_out)
                output = TimeDistributed(Dense(y.shape[-1], activation="linear", name="output"))(tcn_out)
            # For one-shot prediction
            elif dim_y == 2:
                tcn_out = TCN(nb_filters=self.h_dim, dilations=dilations, return_sequences=False, use_skip_connections=False)(tcn_out)
                output = Dense(y.shape[-1], activation="linear", name="output")(tcn_out)
            # Model define and compile
            tcn_model = Model(inputs=[inputs], outputs=[output])
            tcn_model.compile(loss=mse_loss, optimizer=self.adam, metrics=["mse"])

        return tcn_model

    def fit(self, dataset, fold=0, train_split="train", valid_split="val"):
        """Fit the temporal CNN model.
        
        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Cross validation fold
            - train_split: training set splitting parameter
            - valid_split: validation set splitting parameter
            
        Returns:
            - self.tcn_model: trained temporal CNN model
        """
        train_x, train_y = self._data_preprocess(dataset, fold, train_split)
        valid_x, valid_y = self._data_preprocess(dataset, fold, valid_split)

        self.tcn_model = self._build_model(train_x, train_y)

        # Callback for the best model saving
        save_best = ModelCheckpoint(
            self.save_file_name, monitor="val_loss", mode="min", verbose=False, save_best_only=True
        )

        # Train the model
        self.tcn_model.fit(
            train_x,
            train_y,
            batch_size=self.batch_size,
            epochs=self.epoch,
            validation_data=(valid_x, valid_y),
            callbacks=[save_best],
            verbose=self.verbose,
        )

        self.tcn_model.load_weights(self.save_file_name)
        os.remove(self.save_file_name)

        return self.tcn_model

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
        test_y_hat = self.tcn_model.predict(test_x)
        return test_y_hat

    def save_model(self, model_path):
        """Save the model to model_path
        
        Args:
            - model_path: path of the saved model (it should be .h5)
        """
        assert model_path[-3:] == ".h5"
        self.tcn_model.save(model_path)

    def load_model(self, model_path):
        """Load and return the model from model_path        
        
        Args:
            - model_path: path of the saved model (it should be .h5)
        """
        assert model_path[-3:] == ".h5"
        assert os.path.exists(model_path) is True

        loaded_model = tf.keras.models.load_model(model_path, compile=False, custom_objects={"TCN": TCN})

        if self.task == "classification":
            loaded_model.compile(loss=binary_cross_entropy_loss, optimizer=self.adam)
        elif self.task == "regression":
            loaded_model.compile(loss=mse_loss, optimizer=self.adam)

        self.tcn_model = loaded_model
        return loaded_model
