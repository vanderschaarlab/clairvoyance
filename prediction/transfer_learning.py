"""Transfer learning framework.

Using the pre-trained model as the base network, 
fine-tuning the upper model using the transfer dataset.
"""

# Necessary packages
import os
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint
from utils import binary_cross_entropy_loss, mse_loss, rnn_sequential
from base import BaseEstimator, PredictorMixin


class TransferLearning(BaseEstimator, PredictorMixin):
    """Transfer learning model for time-series.
    
    Attributes:
        - predictor_class: trained pre-trained predictive model
        - model_parameters:
            - model_type: 'rnn', 'lstm', or 'gru'
            - h_dim: hidden dimensions
            - n_layer: the number of layers
            - batch_size: the number of samples in each batch
            - epoch: the number of iteration epochs     
            - learning_rate: the number of iteration epochs
            - verbose: print the intermediate results
        - task: classification or regression
    """

    def __init__(self, predictor_class, model_parameters, task):

        self.model_parameters = model_parameters
        self.task = task

        # Network parameters
        self.model_type = model_parameters["model_type"]
        self.h_dim = model_parameters["h_dim"]
        self.n_layer = model_parameters["n_layer"]
        self.batch_size = model_parameters["batch_size"]
        self.epoch = model_parameters["epoch"]
        self.learning_rate = model_parameters["learning_rate"]
        self.verbose = model_parameters["verbose"]

        # Transfer learning on RNN model (where intermediate states exist)
        if model_parameters["model_type"] is not None:
            assert model_parameters["model_type"] in ["rnn", "lstm", "gru"]

        # Model define
        self.predictor_class = predictor_class
        self.transfer_model = None

        # Set path for model saving
        model_path = "tmp"
        model_id = "transfer"

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.save_file_name = "{}/{}".format(model_path, model_id) + datetime.now().strftime("%H%M%S") + ".hdf5"

    def _data_preprocess(self, dataset, fold, split):
        """Preprocess the dataset. Returns feature and label.
        
        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Cross validation fold
            - split: 'train', 'valid' or 'test'
            
        Returns:
            - x: temporal feature
            - y: labels
        """
        # Set temporal, static, label, time, and treatment information
        x, _, y, _, _ = dataset.get_fold(fold, split)

        return x, y

    def _build_model(self, x, y):
        """Construct the transfer learning model using feature and label stats.
        
        Args:
            - x: temporal feature
            - y: labels
            
        Returns:
            - model: transfer learning model
        """
        # Parameters
        dim_y = len(y.shape)

        # Model initialization
        model = tf.keras.Sequential()
        adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

        # For one-shot prediction, use MLP
        if dim_y == 2:
            for _ in range(self.n_layer - 1):
                model.add(layers.Dense(self.h_dim, activation="sigmoid"))

        # For online prediction, use time-series model
        elif dim_y == 3:
            for _ in range(self.n_layer - 1):
                model = rnn_sequential(model, self.model_type, self.h_dim, return_seq=True)

        # For classification
        if self.task == "classification":
            if dim_y == 3:
                model.add(layers.TimeDistributed(layers.Dense(y.shape[-1], activation="sigmoid")))
            elif dim_y == 2:
                model.add(layers.Dense(y.shape[-1], activation="sigmoid"))
            model.compile(loss=binary_cross_entropy_loss, optimizer=adam)

        # For regression
        elif self.task == "regression":
            if dim_y == 3:
                model.add(layers.TimeDistributed(layers.Dense(y.shape[-1], activation="linear")))
            elif dim_y == 2:
                model.add(layers.Dense(y.shape[-1], activation="linear"))
            model.compile(loss=mse_loss, optimizer=adam, metrics=["mse"])

        return model

    def fit(self, dataset, fold=0, train_split="train", valid_split="val"):
        """Fit the transfer learning model.
        
        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Cross validation fold
            - train_split: training set splitting parameter
            - valid_split: validation set splitting parameter
            
        Returns:
            - self.transfer_model: trained transfer model
        """
        # Get intermediate states from the trained predictor model
        intermediate_states = self.predictor_class.get_states(dataset)

        # Set intermediate states as the temporal feature
        ori_temporal_feature = dataset.temporal_feature.copy()
        dataset.temporal_feature = intermediate_states

        # Set training and validation data
        train_x, train_y = self._data_preprocess(dataset, fold, train_split)
        valid_x, valid_y = self._data_preprocess(dataset, fold, valid_split)

        # Build transfer model
        self.transfer_model = self._build_model(train_x, train_y)

        # Callback for the best model saving
        save_best = ModelCheckpoint(
            self.save_file_name, monitor="val_loss", mode="min", verbose=False, save_best_only=True
        )

        # Train the model
        self.transfer_model.fit(
            train_x,
            train_y,
            batch_size=self.batch_size,
            epochs=self.epoch * 10,  # Need more epochs
            validation_data=(valid_x, valid_y),
            callbacks=[save_best],
            verbose=self.verbose,
        )

        # Load the best model
        self.transfer_model.load_weights(self.save_file_name)
        os.remove(self.save_file_name)

        # Restore the original temporal feature
        dataset.temporal_feature = ori_temporal_feature

        return self.transfer_model

    def predict(self, dataset, fold=0, test_split="test"):
        """Return the predictions from the trained model.
        
        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Cross validation fold
            - test_split: testing set splitting parameter
            
        Returns:
            - test_y_hat: predictions
        """
        test_s = self.predictor_class.get_states(dataset, split=test_split)
        test_y_hat = self.transfer_model.predict(test_s)
        return test_y_hat

    def save_model(self, model_path):
        """Save the model to model_path
        
        Args:
            - model_path: path of the saved model (it should be .h5)
        """
        assert model_path[-3:] == ".h5"
        self.transfer_model.save(model_path)

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

        self.transfer_model = loaded_model
        return loaded_model
