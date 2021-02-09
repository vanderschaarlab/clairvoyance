"""TGAIN Imputation.

Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets," ICML, 2018.
"""

# Necessary packages
import numpy as np
from base import BaseEstimator, DataPreprocessorMixin
from imputation.imputation_utils import rounding
from imputation.static.gain_imputation import GainImputation


class TGainImputation(BaseEstimator, DataPreprocessorMixin):
    """TGAIN imputation module.
    
    Attributes:
        - file_name: file_name for saving the trained model
    """

    def __init__(self, file_name):
        # File name for saving the trained model
        assert file_name is not None
        self.file_name = file_name
        # Gain module initialization
        self.gain_model = None

    def fit(self, data):
        """Fit TGAIN model.
        
        Args:
            - data: incomplete 3d numpy array
        """
        # Data preprocessing
        no, time, dim = data.shape
        data_concat = np.reshape(data, [no * time, dim])
        # Only with non-padded data
        idx = np.where(data_concat[:, 1] != -1)[0]

        # Apply TGAIN
        self.gain_model = GainImputation(self.file_name)
        self.gain_model.fit(data_concat[idx, :])
        return

    def transform(self, data):
        """Return imputed data by trained TGAIN model.
        
        Args:
            - data: incomplete 3d numpy array
            
        Returns:
            - data_imputed: complete 3d numpy array after imputation
        """
        # Only after fitting
        assert self.gain_model is not None

        # Data preprocessing
        no, time, dim = data.shape
        data_concat = np.reshape(data, [no * time, dim])
        # Only with non-padded data
        idx = np.where(data_concat[:, 1] != -1)[0]

        # Apply TGAIN
        data_imputed = data_concat.copy()
        data_imputed[idx, :] = self.gain_model.transform(data_concat[idx, :])

        # Rounding
        data_imputed = rounding(data_concat, data_imputed)

        # Return to 3d array
        data_imputed = np.reshape(data_imputed, [no, time, dim])

        return data_imputed
