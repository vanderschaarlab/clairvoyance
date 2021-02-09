"""Utility functions for imputation modules.

(1) rounding: Rounding categorical variables after imputation.
"""

# Necessary packages
import numpy as np


def rounding(data, data_imputed):
    """Use rounding for categorical variables.
    
    Args:
        - data: incomplete original data
        - data_imputed: complete imputed data
        
    Returns:
        - data_imputed: imputed data after rounding
    """
    for i in range(data.shape[1]):
        # If the feature is categorical (category < 20)
        if len(np.unique(data[:, i])) < 20:
            # If values are integer
            if sum(np.round(data[:, i]) == data[:, i]) == len(data[:, i]):
                # Rounding
                data_imputed[:, i] = np.round(data_imputed[:, i])

    return data_imputed
