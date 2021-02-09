"""Interpolation on temporal data

- modes: 'linear', 'quadratic', 'cubic', 'spline'
- Reference: pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
"""

# Necessary packages
import numpy as np
import pandas as pd
from tqdm import tqdm


def interpolation(x, t, imputation_model_name):
    """Interpolate temporal features.

    Args:
        x: temporal features to be interpolated
        t: time information
        imputation_model_name: temporal imputation model (e.g. linear, quadratic, cubic, spline)

    Returns:
        x: interpolated temporal features
    """
    # Parameters
    no, seq_len, dim = x.shape

    # For each patient temporal feature
    for i in tqdm(range(no)):

        temp_x = x[i, :, :]
        temp_t = t[i, :, 0]
        # Only for non-padded data
        idx_x = np.where(temp_x[:, 0] != -1)[0]

        temp_t_hat = temp_t[idx_x]

        # (1) Linear
        if imputation_model_name == "linear":
            temp_x_hat = temp_x[idx_x, :]
            # Convert data type to Dataframe
            temp_x_hat = pd.DataFrame(temp_x_hat)
            # Set time to index for interpolation
            temp_x_hat.index = temp_t_hat

            temp_x_hat = temp_x_hat.interpolate(method=imputation_model_name, limit_direction="both")
            x[i, idx_x, :] = np.asarray(temp_x_hat)

        # (2) Spline, Quadratic, Cubic
        elif imputation_model_name in ["spline", "quadratic", "cubic"]:

            for j in range(dim):
                temp_x_hat = temp_x[idx_x, j]
                # Convert data type to Dataframe
                temp_x_hat = pd.DataFrame(temp_x_hat)
                # Set time to index for interpolation
                temp_x_hat.index = temp_t_hat
                # Interpolate missing values
                # Spline
                if imputation_model_name == "spline":
                    if len(idx_x) - temp_x_hat.isna().sum()[0] > 3:
                        temp_x_hat = temp_x_hat.interpolate(
                            method=imputation_model_name, order=3, fill_value="extrapolate"
                        )
                    else:
                        temp_x_hat = temp_x_hat.interpolate(method="linear", limit_direction="both")
                # Quadratic
                elif imputation_model_name == "quadratic":
                    if len(idx_x) - temp_x_hat.isna().sum()[0] > 2:
                        temp_x_hat = temp_x_hat.interpolate(method=imputation_model_name, fill_value="extrapolate")
                    else:
                        temp_x_hat = temp_x_hat.interpolate(method="linear", limit_direction="both")
                # Cubic
                elif imputation_model_name == "cubic":
                    if len(idx_x) - temp_x_hat.isna().sum()[0] > 3:
                        temp_x_hat = temp_x_hat.interpolate(method=imputation_model_name, fill_value="extrapolate")
                    else:
                        temp_x_hat = temp_x_hat.interpolate(method="linear", limit_direction="both")

                x[i, idx_x, j] = np.asarray(temp_x_hat)[0]

    return x
