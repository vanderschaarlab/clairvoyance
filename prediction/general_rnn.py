"""General RNN modules.

- RNN blocks for classification and regression.
- Different model: Simple RNN, GRU, LSTM
- Regularization: Save the best model
"""

# Necessary packages
import os
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint
from utils import binary_cross_entropy_loss, mse_loss, rnn_sequential
from base import BaseEstimator, PredictorMixin


class GeneralRNN(BaseEstimator, PredictorMixin):
    """RNN predictive model for time-series.
    
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
        model_type_id=None,
        h_dim=None,
        n_layer=None,
        batch_size=None,
        epoch=None,
        learning_rate=None,
        static_mode=None,
        time_mode=None,
        model_id="general_rnn_model",
        model_path="tmp",
        verbose=False,
    ):

        super().__init__(task)

        model_list = ["rnn", "lstm", "gru"]

        if model_type is not None:
            assert model_type in model_list

        if model_type_id is not None:
            # override model_type with id
            assert model_type_id in [0, 1, 2]
            model_type = model_list[model_type_id]

        self.model_type_id = model_type_id

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
        self.predictor_model = None
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
            - a new GeneralRNN
        """
        return GeneralRNN(
            self.task,
            self.model_type,
            self.model_type_id,
            self.h_dim,
            self.n_layer,
            self.batch_size,
            self.epoch,
            self.learning_rate,
            self.static_mode,
            self.time_mode,
            model_id,
            self.model_path,
        )

    def _build_model(self, x, y):
        """Construct the predictive model using feature and label statistics.
        
        Args:
            - x: temporal feature
            - y: labels
            
        Returns:
            - model: predictor model
        """
        # Parameters
        dim = len(x[0, 0, :])
        max_seq_len = len(x[0, :, 0])

        model = tf.keras.Sequential()
        model.add(layers.Masking(mask_value=-1.0, input_shape=(max_seq_len, dim)))

        # Stack multiple layers
        for _ in range(self.n_layer - 1):
            model = rnn_sequential(model, self.model_type, self.h_dim, return_seq=True)

        dim_y = len(y.shape)
        if dim_y == 2:
            return_seq_bool = False
        elif dim_y == 3:
            return_seq_bool = True
        else:
            raise ValueError("Dimension of y {} is not 2 or 3.".format(str(dim_y)))

        model = rnn_sequential(model, self.model_type, self.h_dim, return_seq_bool, name="intermediate_state")
        self.adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

        if self.task == "classification":
            if dim_y == 3:
                model.add(layers.TimeDistributed(layers.Dense(y.shape[-1], activation="sigmoid")))
            elif dim_y == 2:
                model.add(layers.Dense(y.shape[-1], activation="sigmoid"))
            model.compile(loss=binary_cross_entropy_loss, optimizer=self.adam)
        elif self.task == "regression":
            if dim_y == 3:
                model.add(layers.TimeDistributed(layers.Dense(y.shape[-1], activation="linear")))
            elif dim_y == 2:
                model.add(layers.Dense(y.shape[-1], activation="linear"))
            model.compile(loss=mse_loss, optimizer=self.adam, metrics=["mse"])

        return model

    def fit(self, dataset, fold=0, train_split="train", valid_split="val"):
        """Fit the RNN predictor model.
        
        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Cross validation fold
            - train_split: training set splitting parameter
            - valid_split: validation set splitting parameter
            
        Returns:
            - self.predictor_model: trained predictor model
        """
        # Train / Valid dataset
        train_x, train_y = self._data_preprocess(dataset, fold, train_split)
        valid_x, valid_y = self._data_preprocess(dataset, fold, valid_split)

        # Build RNN predictor model
        self.predictor_model = self._build_model(train_x, train_y)

        # Callback for the best model saving
        save_best = ModelCheckpoint(
            self.save_file_name, monitor="val_loss", mode="min", verbose=False, save_best_only=True
        )

        # Train the model
        self.predictor_model.fit(
            train_x,
            train_y,
            batch_size=self.batch_size,
            epochs=self.epoch,
            validation_data=(valid_x, valid_y),
            callbacks=[save_best],
            verbose=self.verbose,
        )

        self.predictor_model.load_weights(self.save_file_name)
        os.remove(self.save_file_name)

        return self.predictor_model

    def predict(self, dataset, fold=0, test_split="test"):
        """Return the predictions based on the trained model.
        
        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Cross validation fold
            - test_split: testing set splitting parameter
            
        Returns:
            - test_y_hat: predictions on testing set
        """
        test_x, _ = self._data_preprocess(dataset, fold, test_split)
        test_y_hat = self.predictor_model.predict(test_x)
        return test_y_hat

    def get_states(self, dataset, fold=0, split="all"):
        """Returns the intermediate state of the predictor for transfer learning.
        
        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: data fold number
            - split: always 'all'
            
        Returns:
            - data_s: intermediate_state for the input data
        """

        data_x, _ = self._data_preprocess(dataset, fold, split)

        layer_name = "intermediate_state"
        intermediate_model = tf.keras.Model(
            inputs=self.predictor_model.input, outputs=self.predictor_model.get_layer(layer_name).output
        )
        data_s = intermediate_model.predict(data_x)
        return data_s

    def save_model(self, model_path):
        """Save the model to model_path
        
        Args:
            - model_path: path of the saved model (it should be .h5)
        """
        assert model_path[-3:] == ".h5"
        self.predictor_model.save(model_path)

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

        self.predictor_model = loaded_model
        return loaded_model

    @staticmethod
    def get_hyperparameter_space():
        hyp_ = [
            {"name": "h_dim", "type": "discrete", "domain": list(range(10, 150, 10)), "dimensionality": 1},
            {"name": "n_layer", "type": "discrete", "domain": list(range(1, 4, 1)), "dimensionality": 1},
            {"name": "batch_size", "type": "discrete", "domain": list(range(100, 1001, 100)), "dimensionality": 1},
            # NOTE: Comment out the below line if don't want to sweep through all types of GeneralRNN at once:
            {"name": "model_type_id", "type": "discrete", "domain": [0, 1, 2], "dimensionality": 1},
            {"name": "learning_rate", "type": "continuous", "domain": [0.0005, 0.01], "dimensionality": 1},
        ]

        return hyp_
