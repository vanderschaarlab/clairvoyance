"""MRNN Imputation.

Reference: Jinsung Yoon, William R. Zame and Mihaela van der Schaar, 
                     "Estimating Missing Data in Temporal Data Streams Using Multi-Directional Recurrent Neural Networks," 
                     in IEEE Transactions on Biomedical Engineering, vol. 66, no. 5, pp. 1477-1490, May 2019.
"""

# Necessary packages
import numpy as np
import os
import shutil
from base import BaseEstimator, DataPreprocessorMixin
from imputation.imputation_utils import rounding
from imputation.temporal.mrnn.mrnn import MRnn


class MRnnImputation(BaseEstimator, DataPreprocessorMixin):
    """MRNN imputation module.
    
    Attributes:
        - file_name: file_name for saving the trained model
    """

    def __init__(self, file_name):
        # File name for saving the trained model
        assert file_name is not None
        self.save_file_directory = "tmp/" + file_name
        # Median values for median imputation
        self.median_vals = None
        # MRNN model
        self.mrnn_model = None

    def compute_median_value(self, data):
        """Compute median without nan and padded values.
        
        Args:
            - data: incomplete data
            
        Returns:
            - median_vals: median of each variable
        """
        new_data = data.copy()
        # For temporal features
        if len(data.shape) == 3:
            new_data = np.reshape(new_data, [data.shape[0] * data.shape[1], data.shape[2]])
        # Excluding padded values
        idx = np.where(new_data[:, 0] != -1)[0]
        # Compute median of the data
        median_vals = np.nanmedian(new_data[idx, :], axis=0)
        return median_vals

    def mrnn_data_preprocess(self, data):
        """Transform data to x, m, t, and data_new for MRNN training and testing.
        
        Args:
            - data: original data
            
        Returns:
            - x: temporal features
            - m: mask matrix
            - t: time information
            - data_new: fill na to 0 on df
            - median_vals: median values
        """
        # Compute median_value
        median_vals = self.compute_median_value(data.copy())
        # mask matrix
        m = 1 - np.isnan(data)
        # Replace nan to 0
        data = np.nan_to_num(data, 0)
        # Save data without nan
        data_new = data.copy()
        # Replace -1 to 0
        idx = np.where(data == -1)
        data[idx] = 0
        x = data.copy()

        no, seq_len, dim = data.shape

        # Compute t
        t = list()
        for i in range(no):
            temp_t = np.ones([seq_len, dim])
            for j in range(dim):
                for k in range(1, seq_len):
                    if m[i, k, j] == 0:
                        temp_t[k, j] = temp_t[k - 1, j] + 1
            t = t + [temp_t]

        t = np.asarray(t)

        return x, m, t, data_new, median_vals

    def fit(self, data):
        """Fit MRNN model.
        
        Args:
            - data: incomplete dataset
        """
        mrnn_model_parameters = {"h_dim": 10, "batch_size": 128, "iteration": 2000, "learning_rate": 0.01}

        # Reset the directory for saving
        if os.path.exists(self.save_file_directory):
            shutil.rmtree(self.save_file_directory)

        # Data preprocess for mrnn
        x, m, t, df_new, self.median_vals = self.mrnn_data_preprocess(data)

        # Fit MRNN model
        self.mrnn_model = MRnn(x, self.save_file_directory)
        self.mrnn_model.fit(x, m, t, self.median_vals, mrnn_model_parameters)

        return

    def transform(self, data):
        """Return imputed data by trained MRNN model.
        
        Args:
            - data: 3d numpy array with missing data
            
        Returns:
            - data_imputed: 3d numpy array with imputed data
        """
        # Only after fit
        assert self.median_vals is not None
        assert self.mrnn_model is not None
        # Data preprocess for mrnn
        x, m, t, data_new, _ = self.mrnn_data_preprocess(data)

        # Impute missing data
        imputed_x = self.mrnn_model.transform(x, m, t, self.median_vals)
        data_imputed = (1 - m) * imputed_x + m * data_new

        # Rounding
        data_imputed = rounding(data_new, data_imputed)

        return data_imputed
