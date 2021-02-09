"""Utility functions for data.

(0) Define root and data directory
(1) concate_xs: Concatenate temporal and static features
(2) concate_xt: Concatenate temporal anf time features
(3) list_diff: compute the difference between two lists in order
(4) padding: put -1 values to the sequences outside of the time range
(5) index_reset: return the pandas dataset with reset indice
(6) pd_list_to_np_array: convert list of pandas to 3d array
(7) normalization: MinMax Normalizer
(8) renormalization: MinMax renormalizer
"""

# Necessary packages
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

# Define root and data directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "../datasets/data/")


def concate_xs(x, s):
    """Concatenate static features to temporal feature for every time point.
    
    Args:
        x: temporal features
        s: static features
        
    Returns:
        concate_x: concatenate temporal and static features
    """
    concate_x = list()

    for i in range(len(s[:, 0])):
        temp_x = x[i]
        temp_s = np.repeat(np.reshape(s[i, :], [1, -1]), len(temp_x[:, 0]), axis=0)
        # -1 padding
        pad_idx = sum(temp_x[:, 0] == -1)
        if pad_idx > 0:
            temp_s[-pad_idx:, :] = -1
        # Concatenate
        temp_xs = np.concatenate((temp_x, temp_s), axis=1)
        concate_x = concate_x + [temp_xs]

    concate_x = np.asarray(concate_x)

    return concate_x


def concate_xt(x, t):
    """Concatenate time feature to temporal feature for every time point.
    
    Args:
        x: temporal features
        t: time feature
        
    Returns:
        concate_x: concatenate temporal and time features
    """
    concate_x = list()

    for i in range(len(t[:, 0, 0])):
        temp_x = x[i]
        temp_t = t[i]
        temp_xt = np.concatenate((temp_x, temp_t), axis=1)
        concate_x = concate_x + [temp_xt]

    concate_x = np.asarray(concate_x)

    return concate_x


def list_diff(list1, list2):
    """Compute list differences in order.
    
    Args:
        - list1: first list
        - list2: second list
        
    Returns:
        - out: list difference
    """
    out = []
    for ele in list1:
        if not ele in list2:
            out.append(ele)

    return out


def padding(x, max_seq_len):
    """Sequence data padding.
    
    Args:
        - x: temporal features
        - max_seq_len: maximum sequence_length
        
    Returns:
        - x_hat: padded temporal features
    """
    # Shape of the temporal features
    seq_len, dim = x.shape
    col_name = x.columns.values

    # Padding (-1)
    x_pad_hat = -np.ones([max_seq_len - seq_len, dim])
    x_pad_hat = pd.DataFrame(x_pad_hat, columns=col_name)

    x_hat = pd.concat((x, x_pad_hat), axis=0)
    x_hat["id"] = np.unique(x["id"])[0]

    x_hat = index_reset(x_hat)

    return x_hat


def index_reset(x):
    """Reset index in the pandas dataframe.
    
    Args:
        x: original pandas dataframe
        
    Returns:
        x: data with new indice
    """
    x = x.reset_index()
    if "index" in x.columns:
        x = x.drop(columns=["index"])

    return x


def pd_list_to_np_array(x, drop_columns):
    """Convert list of pandas dataframes to 3d numpy array.
    
    Args:
        - x: list of pandas dataframe
        - drop_column: column names to drop before converting to numpy array
        
    Returns:
        - x_hat: 3d numpy array
    """
    x_hat = list()
    for component in x:
        temp = component.drop(columns=drop_columns)
        temp = np.asarray(temp)
        x_hat = x_hat + [temp]

    x_hat = np.asarray(x_hat)
    return x_hat


def normalization(df, subtract_val, division_val):
    """Normalizer.
        
    Args:
        - df: input data
        - normalizer_type: 'minmax' or 'standard'
        
    Returns:
        - df: normalized data
        - norm_parameters: parameters for renomalization
    """

    for col_name in df.columns:
        df[col_name] = df[col_name] - subtract_val[col_name]
        df[col_name] = df[col_name] / (division_val[col_name] + 1e-8)

    return df


def get_normalization_param(df, normalizer_type):
    """Normalizer.

        Args:
            - df: input data
            - normalizer_type: 'minmax' or 'standard'

        Returns:
            - df: normalized data
            - norm_parameters: parameters for renomalization
        """
    assert normalizer_type in ["standard", "minmax"]

    if normalizer_type == "standard":
        subtract_val = df.mean()
        division_val = df.std()
    else:
        subtract_val = df.min()
        division_val = df.max() - df.min()

    norm_parameters = {"subtract_val": subtract_val, "division_val": division_val}

    return norm_parameters


def renormalization(df, norm_parameters):
    """Renormalizer.
        
    Args:
        - df: input data
        - norm_parameters: parameters for renomalization
        
    Returns:
        - df: renormalized data
    """
    subtract_val = norm_parameters["subtract_val"]
    division_val = norm_parameters["division_val"]

    feature_names = subtract_val.keys()

    for f in feature_names:
        assert f in df.columns

    for col_name in df.columns:
        df.loc[df.index, col_name] = df.loc[df.index, col_name] - subtract_val[col_name]
        df.loc[df.index, col_name] = df.loc[df.index, col_name] / (division_val[col_name] + 1e-8)

    return df
