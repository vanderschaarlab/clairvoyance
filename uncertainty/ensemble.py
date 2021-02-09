"""Uncertainty estimation with ensemble methods.

Using the prediction variance of different methods as the uncertainty.
"""

# Necessary packages
import numpy as np
import os
import keras
from base import BaseEstimator, PredictorMixin
from prediction import prediction
from utils import list_diff


class EnsembleUncertainty(BaseEstimator, PredictorMixin):
    """Uncertainty estimator by ensemble method.
    
    Attributes:
        - predictor_model: original predictor model
        - ensemble_model_type: types of models for ensemble model
        - task: classification or regression
        - model_type: 'rnn', 'lstm', 'gru', 'attention', 'tcn', 'transformer'
        - h_dim: hidden dimensions
        - n_head: number of head (for transformer model)
        - n_layer: the number of layers
        - batch_size: the number of samples in each batch
        - epoch: the number of iteration epochs
        - learning_rate: learning rates
        - static_mode: 'concatenate' or None
        - time_mode: 'concatenate' or None
    """

    def __init__(
        self,
        predictor_model=None,
        ensemble_model_type=None,
        task=None,
        model_type=None,
        h_dim=None,
        n_head=None,
        n_layer=None,
        batch_size=None,
        epoch=None,
        learning_rate=None,
        static_mode=None,
        time_mode=None,
        verbose=None,
    ):

        super().__init__(predictor_model, ensemble_model_type, task)

        self.predictor_model = predictor_model
        self.ensemble_model_type = ensemble_model_type
        self.task = task

        self.model_type = model_type
        self.h_dim = h_dim
        self.n_head = n_head
        self.n_layer = n_layer
        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.static_mode = static_mode
        self.time_mode = time_mode
        self.verbose = verbose

        # Define ensemble model
        self.ensemble_model = None

    def fit(self, dataset):
        """Fit the ensemble models for uncertainty estimation.
        
        Args:
            - dataset: temporal, static, label, time, treatment information
            
        Returns:
            - self.ensemble_model: trained ensemble model
        """
        # Define model parameters
        self.model_parameters = {
            "h_dim": self.h_dim,
            "n_head": self.n_head,
            "n_layer": self.n_layer,
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "learning_rate": self.learning_rate,
            "model_type": self.model_type,
            "static_mode": self.static_mode,
            "time_mode": self.time_mode,
            "verbose": self.verbose,
        }

        # Define ensemble model type
        if self.model_parameters["model_type"] in self.ensemble_model_type:
            self.ensemble_model_type = list_diff(self.ensemble_model_type, [self.model_parameters["model_type"]])

        # Initialize ensemble model to the currently trained model
        self.ensemble_model = [self.predictor_model]

        for each_model_type in self.ensemble_model_type:
            self.model_parameters["model_type"] = each_model_type
            pred_class = prediction(each_model_type, self.model_parameters, self.task)
            pred_class.fit(dataset)

            self.ensemble_model = self.ensemble_model + [pred_class]

        return

    def predict(self, dataset):
        """Return the confidence intervals.
        
        Args:
            - dataset: temporal, static, label, time, treatment information
            
        Returns:
            - test_ci_hat: uncertainty of each predictions
        """
        all_y_hat = list()

        for each_model in self.ensemble_model:
            test_y_hat = each_model.predict(dataset)
            all_y_hat = all_y_hat + [test_y_hat]

        # Standard deviation of different model predictions
        test_ci_hat = np.std(np.asarray(all_y_hat), axis=0)

        return test_ci_hat

    def save_model(self, model_path):
        """Save the model to model_path.
        
        Args:
            - model_path: directory of the saved model (it should be /)
        """
        assert os.path.isdir is True

        for i in range(len(self.ensemble_model)):
            save_file_name = model_path + "ensemble_model_" + str(i + 1) + ".h5"
            self.ensemble_model[i].save(save_file_name)

    def load_model(self, model_path):
        """Load and return the model from model_path        
        
        Args:
            - model_path: path of the saved model (it should be .h5)
        """
        assert os.path.isdir is True

        file_names = [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]

        loaded_models = list()
        for i in range(len(file_names)):
            assert file_names[i][-3:] == ".h5"
            each_model = keras.models.load_model(file_names[i])
            loaded_models = loaded_models + [each_model]

        return loaded_models
