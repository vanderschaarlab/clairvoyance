"""Deep Sensing Implementation.

Reference: J. Yoon, W. R. Zame, M. van der Schaar, 
"Deep Sensing: Active Sensing using Multi-directional Recurrent Neural Networks," 
ICLR, 2018.
"""

# Necessary packages
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm

from prediction import prediction
from base import BaseEstimator, PredictorMixin


class DeepSensing(BaseEstimator, PredictorMixin):
    """Deep Sensing core functions.
        
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
        model_id="deepsensing_model",
        model_path="tmp",
        verbose=False,
    ):

        super().__init__(task)

        if model_type is not None:
            # DeepSensing model is RNN-based model
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

        # Predictive model with original label
        self.pred_class = None
        self.pred_model = None

        # Deep sensing models
        self.deep_sensing_class = None
        self.deep_sensing_model = None

    def model_train(self, dataset, model_parameters, task):
        """Train predictive model.
        
        Args:
            - dataset: training dataset
            - model_parameters: parameters for training model
            - task: classification or regression
            
        Returns:
            - pred_class: class of predictive model
            - pred_model: trained predictive model
        """
        # Build model
        pred_class = prediction(model_parameters["model_type"], model_parameters, task)
        # Train the model
        pred_model = pred_class.fit(dataset)

        return pred_class, pred_model

    def new_label_generation(self, dataset, split="all"):
        """Label generation for deep sensing.
        
        Args:
            - dataset: training dataset
            - split: use the entire data ('all')
            
        Returns:
            - y_tilde: new labels for training deep sensing model
        """

        # Only for the single label
        assert dataset.label.shape[-1] == 1

        # Get the features and predictions
        x, y = self.pred_class._data_preprocess(dataset, fold=0, split=split)
        y_hat = self.pred_class.predict(dataset, test_split=split)

        # Parameters
        no, seq_len, dim = x.shape

        # For one-shot prediction
        if len(y_hat.shape) == 2:
            # Output initialization
            y_tilde = np.zeros([no, dim])
            # For each sample
            for i in tqdm(range(no)):
                # Only reset the last features
                temp_x = x[i, :, :].copy()
                temp_refer = temp_x[seq_len - 1, :].copy()
                temp_x = np.reshape(temp_x, [1, seq_len, dim])
                temp_x[0, seq_len - 1, :] = 0
                # For each dimensions
                for j in range(dim):
                    # Restore original value for a certain feature
                    temp_x[0, seq_len - 1, j] = temp_refer[j]
                    # See the prediction in comparison to original prediction
                    y_tilde[i, j] = np.abs(self.pred_model.predict(temp_x) - y_hat[i, 0])
                    # Restore the data
                    temp_x[0, seq_len - 1, j] = 0

        # For online prediction
        elif len(y_hat.shape) == 3:
            # Output initialization
            y_tilde = np.zeros([no, seq_len, dim])
            # For each sample
            for i in tqdm(range(no)):
                # Save the reference
                temp_refer = x[i, :, :].copy()
                # -1 padding
                temp_x = -np.ones([1, seq_len, dim])
                temp_x[0, seq_len - 1, :] = 0
                # For each time-point
                for t in range(seq_len):
                    # Set previous measurements
                    temp_x[0, (seq_len - t) :, :] = temp_refer[:t, :]
                    # For each feature
                    for j in range(dim):
                        # Restore original value for a certain feature
                        temp_x[0, seq_len - 1, j] = temp_refer[t, j]
                        # See the prediction in comparison to original prediction
                        y_tilde[i, t, j] = np.abs(self.pred_model.predict(temp_x)[0, seq_len - 1, 0] - y_hat[i, t, 0])
                        # Restore the data
                        temp_x[0, seq_len - 1, j] = 0

        return y_tilde

    def fit(self, dataset):
        """Deep sensing model training.
        
        Args:
            - dataset: training dataset
        """
        # Define model parameters
        self.model_parameters = {
            "h_dim": self.h_dim,
            "n_layer": self.n_layer,
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "model_type": self.model_type,
            "learning_rate": self.learning_rate,
            "static_mode": self.static_mode,
            "time_mode": self.time_mode,
            "verbose": True,
        }

        # Train the model to the original label
        self.pred_class, self.pred_model = self.model_train(dataset, self.model_parameters, self.task)

        # New label generation
        new_y = self.new_label_generation(dataset)

        # Replace original label to new label
        ori_y = dataset.label.copy()
        dataset.label = new_y

        # Train deep sensing model
        self.deep_sensing_class, self.deep_sensing_model = self.model_train(
            dataset, self.model_parameters, task="regression"
        )

        # Restore original label
        dataset.label = ori_y

    def predict(self, dataset, threshold=0.01):
        """Return the active sensing results.
        
        Args:
            - dataset: testing dataset
            - threshold: selection threshold
            
        Returns:
            - selection: selection results by deep sensing 
        """
        y_hat = self.deep_sensing_class.predict(dataset, test_split="test")
        selection = 1 * (y_hat > threshold)

        return selection, y_hat

    def save_model(self, model_path):
        """Save the model to model_path
        
        Args:
            - model_path: path of the saved model (it should be .h5)
        """
        assert model_path[-3:] == ".h5"
        self.deep_sensing_model.save(model_path)

    def load_model(self, model_path):
        """Load and return the model from model_path        
        
        Args:
            - model_path: path of the saved model (it should be .h5)
        """
        assert model_path[-3:] == ".h5"
        assert os.path.exists(model_path) is True

        loaded_model = tf.keras.models.load_model(model_path)
        return loaded_model
