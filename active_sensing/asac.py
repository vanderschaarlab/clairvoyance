"""ASAC: Active Sensing using Actor-Critic models.

Reference: J. Yoon, J. Jordon, M. van der Schaar, 
"ASAC: Active Sensing using Actor-Critic models," MLHC, 2019
"""

# Necessary Packages
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Masking
from tensorflow.keras.layers import TimeDistributed, Multiply, Lambda, Activation
from tensorflow.keras.models import Model

from datetime import datetime
import numpy as np
import os
from utils import concate_xs, concate_xt
from utils import select_loss, rmse_loss, rnn_layer, binary_cross_entropy_loss

from base import BaseEstimator, PredictorMixin


class Asac(BaseEstimator, PredictorMixin):
    """Asac core functions.
    
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
        - verbose: print intermediate states
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
        model_id="asac_model",
        model_path="tmp",
        verbose=False,
    ):

        super().__init__(task)

        if model_type is not None:
            # ASAC model is RNN-based model
            assert model_type in ["rnn", "lstm", "gru"]

        self.task = task
        self.model_type = model_type
        self.h_dim = h_dim
        self.n_layer = n_layer
        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.static_mode = static_mode
        self.time_mode = time_mode
        self.verbose = verbose

        # Set path for model saving
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.save_file_name = "{}/{}".format(model_path, model_id) + datetime.now().strftime("%H%M%S") + ".hdf5"

        # ASAC model define
        self.asac_model = None

    def _data_preprocess(self, dataset, fold, split):
        """Preprocess the dataset.
        
        Returns feature, label and previous features.
        
        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Cross validation fold
            - split: 'train', 'valid' or 'test'
            
        Returns:
            - x: temporal feature
            - y: labels
            - previous_x: previous features
        """
        # Set temporal, static, label, and time information
        x, s, y, t, _ = dataset.get_fold(fold, split)

        # Concatenate static features
        if self.static_mode == "concatenate":
            if s is not None:
                x = concate_xs(x, s)

        # Concatenate time information
        if self.time_mode == "concatenate":
            if t is not None:
                x = concate_xt(x, t)

        # Define previous temporal features (push one time stamp)
        previous_x = x.copy()
        for i in range(len(x)):
            previous_x[i][1:, :] = previous_x[i][:-1, :]
            previous_x[i][0, :] = -np.ones([len(x[i][0, :])])

        return x, y, previous_x

    def _build_model(self, x, y):
        """Construct ASAC model using feature and label statistics.
        
        Args:
            - x: temporal feature
            - y: labels
            
        Returns:
            - model: asac model
        """

        # Parameters
        h_dim = self.h_dim
        n_layer = self.n_layer
        dim = len(x[0, 0, :])
        max_seq_len = len(x[0, :, 0])

        # Build one input, two outputs model
        main_input = Input(shape=(max_seq_len, dim), dtype="float32")
        mask_layer = Masking(mask_value=-1.0)(main_input)
        previous_input = Input(shape=(max_seq_len, dim), dtype="float32")
        previous_mask_layer = Masking(mask_value=-1.0)(previous_input)

        select_layer = rnn_layer(previous_mask_layer, self.model_type, h_dim, return_seq=True)
        for _ in range(n_layer):
            select_layer = rnn_layer(select_layer, self.model_type, h_dim, return_seq=True)
        select_layer = TimeDistributed(Dense(dim, activation="sigmoid"))(select_layer)

        # Sampling the selection
        select_layer = Lambda(lambda x: x - 0.5)(select_layer)
        select_layer = Activation("relu")(select_layer)
        select_out = Lambda(lambda x: x * 2, name="select")(select_layer)

        # Second output
        pred_layer = Multiply()([mask_layer, select_out])

        for _ in range(n_layer - 1):
            pred_layer = rnn_layer(pred_layer, self.model_type, h_dim, return_seq=True)

        return_seq_bool = len(y.shape) == 3
        pred_layer = rnn_layer(pred_layer, self.model_type, h_dim, return_seq_bool)

        if self.task == "classification":
            act_fn = "sigmoid"
        elif self.task == "regression":
            act_fn = "linear"

        if len(y.shape) == 3:
            pred_out = TimeDistributed(Dense(y.shape[-1], activation=act_fn), name="predict")(pred_layer)
        elif len(y.shape) == 2:
            pred_out = Dense(y.shape[-1], activation=act_fn, name="predict")(pred_layer)

        model = Model(inputs=[main_input, previous_input], outputs=[select_out, pred_out])
        # Optimizer
        adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
        # Model compile
        if self.task == "classification":
            model.compile(
                loss={"select": select_loss, "predict": binary_cross_entropy_loss},
                optimizer=adam,
                loss_weights={"select": 0.01, "predict": 1},
            )
        elif self.task == "regression":
            model.compile(
                loss={"select": select_loss, "predict": rmse_loss},
                optimizer=adam,
                loss_weights={"select": 0.01, "predict": 1},
            )

        return model

    def fit(self, dataset, fold=0, train_split="train", valid_split="val"):
        """Fit ASAC model.
        
        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Cross validation fold
            - train_split: training set splitting parameter
            - valid_split: validation set splitting parameter
        """
        # Data preprocessing
        train_x, train_y, previous_train_x = self._data_preprocess(dataset, fold, train_split)
        valid_x, valid_y, previous_valid_x = self._data_preprocess(dataset, fold, valid_split)

        # Model defining
        self.asac_model = self._build_model(train_x, train_y)

        # Callback for the best model saving
        save_best = ModelCheckpoint(
            self.save_file_name, monitor="val_loss", mode="min", verbose=False, save_best_only=True
        )

        # Train the model
        self.asac_model.fit(
            [train_x, previous_train_x],
            [train_x, train_y],
            batch_size=self.batch_size,
            epochs=self.epoch,
            validation_data=([valid_x, previous_valid_x], [valid_x, valid_y]),
            callbacks=[save_best],
            verbose=self.verbose,
        )
        # Load the best model
        self.asac_model.load_weights(self.save_file_name)
        os.remove(self.save_file_name)

        return

    def predict(self, dataset, fold=0, test_split="test"):
        """Return the temporal and feature importance for future measurements.
        
        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Cross validation fold
            - test_split: testing set splitting parameter
            
        Returns:
            - test_s_hat: next measurement recommendation
        """
        # Data preprocessing
        test_x, _, second_test_x = self._data_preprocess(dataset, fold, test_split)
        # Model prediction
        test_s_hat, _ = self.asac_model.predict([test_x, second_test_x])
        return np.round(test_s_hat), None

    def save_model(self, model_path):
        """Save the model to model_path
        
        Args:
            - model_path: path of the saved model (it should be .h5)
        """
        assert model_path[-3:] == ".h5"
        self.asac_model.save(model_path)

    def load_model(self, model_path):
        """Load and return the model from model_path        
        
        Args:
            - model_path: path of the saved model (it should be .h5)
        """
        assert model_path[-3:] == ".h5"
        assert os.path.exists(model_path) is True

        loaded_model = tf.keras.models.load_model(model_path, compile=False)

        # Compile with user-defined loss
        if self.task == "classification":
            loaded_model.compile(loss=binary_cross_entropy_loss, optimizer=self.adam)
        elif self.task == "regression":
            loaded_model.compile(loss=rmse_loss, optimizer=self.adam)

        self.asac_model = loaded_model
        return loaded_model
