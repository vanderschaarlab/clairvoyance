"""General RNN modules.

- RNN blocks for classification and regression.
- Different model: Simple RNN, GRU, LSTM
- Regularization: Save the best model
"""

# Necessary packages
import os
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from datetime import datetime
from utils import binary_cross_entropy_loss, mse_loss
from base import BaseEstimator, PredictorMixin
import pickle
import numpy as np


class GBM(BaseEstimator, PredictorMixin):
    """RNN predictive model for time-series.

    Attributes:
        - task: classification or regression
        - learning_rate: the learning rate
        - n_estimators: number of trees
        - max_depth: max tree depth

        - static_mode: 'concatenate' or None
        - time_mode: 'concatenate' or None
        - model_id: the name of model
        - model_path: model path for saving
        - verbose: print intermediate process
    """

    def __init__(
        self,
        task=None,
        n_estimators=None,
        max_depth=None,
        learning_rate=None,
        static_mode=None,
        time_mode=None,
        model_id="stationary_gbm",
        model_path="tmp",
        verbose=False,
    ):

        super().__init__(task)

        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.static_mode = static_mode
        self.time_mode = time_mode
        self.model_path = model_path
        self.model_id = model_id
        self.verbose = verbose

        # Predictor model & optimizer define
        self.predictor_model = None

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
        return GBM(
            self.task,
            self.n_estimators,
            self.max_depth,
            self.learning_rate,
            self.static_mode,
            self.time_mode,
            model_id,
            self.model_path,
            self.verbose,
        )

    def format_stationary_data(self, x, y, task, mode="train"):
        # N_sample, max_seq_len, dim

        dim_y = len(y.shape)
        if dim_y == 2:
            raise NotImplementedError
        elif dim_y == 3:
            return_seq_bool = True
        else:
            raise ValueError("Dimension of y {} is not 2 or 3.".format(str(dim_y)))

        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        y = y.reshape(y.shape[0] * y.shape[1])

        if mode == "train":
            # filter out -1
            ind = ~(y == -1)
            loc = np.argwhere(y == -1)
            y = y[ind]
            x = np.delete(x, loc, axis=0)

        assert x.shape[0] == y.shape[0]
        if task == "classification":
            y = (~(y == 0)).astype(np.int)
        return x, y

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

        train_x, train_y = self.format_stationary_data(train_x, train_y, self.task)
        valid_x, valid_y = self.format_stationary_data(valid_x, valid_y, self.task)

        # Build RNN predictor model
        if self.task == "classification":
            self.predictor_model = GradientBoostingClassifier(
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                max_features="sqrt",
            )
        elif self.task == "regression":
            self.predictor_model = GradientBoostingRegressor(
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                max_features="sqrt",
            )

        # Train the model
        self.predictor_model.fit(train_x, train_y)

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
        test_x, test_y = self._data_preprocess(dataset, fold, test_split)
        shape0 = test_y.shape[0]
        shape1 = test_y.shape[1]
        shape2 = test_y.shape[2]

        test_x, test_y = self.format_stationary_data(test_x, test_y, self.task, "test")

        if self.task == "regression":
            test_y_hat = self.predictor_model.predict(test_x)
        else:
            # not implemented for multiclass classification
            test_y_hat = self.predictor_model.predict_proba(test_x)[:, 1]
        test_y_hat = test_y_hat.reshape(shape0, shape1, shape2)
        return test_y_hat

    def save_model(self, model_path):
        """Save the model to model_path

        Args:
            - model_path: path of the saved model
        """
        with open(model_path, "wb") as f:
            pickle.dump(self.predictor_model, f)

    def load_model(self, model_path):
        """Load and return the model from model_path

        Args:
            - model_path: path of the saved model
        """
        assert os.path.exists(model_path) is True

        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)

        self.predictor_model = loaded_model
        return loaded_model

    @staticmethod
    def get_hyperparameter_space():
        hyp_ = [
            {"name": "n_estimators", "type": "discrete", "domain": list(range(100, 1001, 100)), "dimensionality": 1},
            {"name": "max_depth", "type": "discrete", "domain": list(range(2, 8, 1)), "dimensionality": 1},
            {"name": "learning_rate", "type": "continuous", "domain": [0.0005, 0.1], "dimensionality": 1},
        ]
        return hyp_
