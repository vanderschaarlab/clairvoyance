"""Data preprocessing: Encoding.

(1) OneHotEncoder: Convert categorical variable to one-hot encoded variable
(2) MinMaxNormalizer: Normalizer to make the feature range within [0, 1]
(3) StandardNormalizer: Normalizer with mean = 0, std = 1 criteria
(4) ReNormalizer: Recover the original data for visualization
(5) Normalizer: Combine MinMaxNormalizer and StandardNormalizer
(6) ProblemMaker: Define temporal, static, label, treatment and time features
"""

# Necessary packages
import pandas as pd
import numpy as np
from tqdm import tqdm
from base import BaseEstimator, DataPreprocessorMixin
from utils.data_utils import normalization, renormalization, padding, get_normalization_param
from utils.data_utils import list_diff, index_reset, pd_list_to_np_array


class OneHotEncoder(BaseEstimator, DataPreprocessorMixin):
    """Return one-hot encoded dataset.
    
    Attributes:
        - one_hot_encoding_feature: features that need one-hot encoding.
    """

    def __init__(self, one_hot_encoding_features):
        self.one_hot_encoding_features = one_hot_encoding_features

    def fit(self, dataset):
        pass

    def transform(self, dataset):
        """Transform original dataset to one-hot encoded data.
        
        Args:
            - dataset: original PandasDataset
        
        Returns:
            - dataset: one-hot encoded PandasDataset
        """
        if self.one_hot_encoding_features is not None:
            # For each feature
            for feature_name in self.one_hot_encoding_features:
                # Temporal features
                if dataset.temporal_data is not None and feature_name in dataset.temporal_data.columns:
                    dataset.temporal_data = pd.get_dummies(dataset.temporal_data, columns=[feature_name])
                # Static features
                elif dataset.static_data is not None and feature_name in dataset.static_data.columns:
                    dataset.static_data = pd.get_dummies(dataset.static_data, columns=[feature_name])

        return dataset

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)


class MinMaxNormalizer(BaseEstimator, DataPreprocessorMixin):
    """Normalize the data to make the range within [0, 1].
    """

    def __init__(self):
        self.norm_parameters = None
        self._normalizer_type = "minmax"

    def fit(self, dataset):
        # For temporal data
        x = dataset.temporal_data
        if x is not None:
            temporal_col_names = x.drop(["id", "time"], axis=1).columns.values
            temporal_norm_parameters = get_normalization_param(
                x[temporal_col_names], normalizer_type=self._normalizer_type
            )
        # For static data
        s = dataset.static_data
        if s is not None:
            static_col_names = s.drop(["id"], axis=1).columns.values
            static_norm_parameters = get_normalization_param(s[static_col_names], normalizer_type=self._normalizer_type)
        # Normalization parameters for renomalization
        self.norm_parameters = {
            "normalizer": self._normalizer_type,
            "temporal": temporal_norm_parameters,
            "static": static_norm_parameters,
        }

    def transform(self, dataset):
        """Transform original dataset to MinMax normalized dataset.

        Args:
            - dataset: original PandasDataset

        Returns:
            - dataset: normalized PandasDataset
            - norm_parameters: normalization parameters for renomalization
        """
        # For temporal data
        x = dataset.temporal_data
        if x is not None:
            temporal_col_names = x.drop(["id", "time"], axis=1).columns.values
            x[temporal_col_names] = normalization(x[temporal_col_names], **self.norm_parameters["temporal"])
        # For static data
        s = dataset.static_data
        if s is not None:
            static_col_names = s.drop(["id"], axis=1).columns.values
            s[static_col_names] = normalization(s[static_col_names], **self.norm_parameters["static"])
        return dataset

    def fit_transform(self, dataset):
        """Transform original dataset to MinMax normalized dataset.
        
        Args:
            - dataset: original PandasDataset
        
        Returns:
            - dataset: normalized PandasDataset
            - norm_parameters: normalization parameters for renomalization
        """
        self.fit(dataset)
        return self.transform(dataset)


class StandardNormalizer(BaseEstimator, DataPreprocessorMixin):
    """Normalize the data to make mean = 0 and std = 1.    
        
    Very similar to MinMaxNormalizer. 
    """

    def __init__(self):
        self.norm_parameters = None
        self.norm = MinMaxNormalizer()
        self.norm.normalizer_type = "standard"

    def fit(self, dataset):
        self.norm.fit(dataset)

    def transform(self, dataset):
        return self.norm.transform(dataset)

    def fit_transform(self, dataset):
        """Transform original dataset to standard normalized dataset.
        
        Args:
            - dataset: original PandasDataset
        
        Returns:
            - dataset: normalized PandasDataset
            - norm_parameters: normalization parameters for renomalization
        """
        self.fit(dataset)
        return self.transform(dataset)


class ReNormalizer(BaseEstimator, DataPreprocessorMixin):
    """Recover the original data from normalized data.
    
    Attributes:
        - norm_parameters: normalization parameters for renomalization
    """

    def __init__(self, norm_parameters):
        self.temporal_norm_parameters = norm_parameters["temporal"]
        self.static_norm_parameters = norm_parameters["static"]

    def fit(self, dataset):
        pass

    def transform(self, dataset):
        return self.fit_transform(dataset)

    def fit_transform(self, dataset):
        """Transform normalized dataset to original dataset.
        
        Args:
            - dataset: normalized PandasDataset
        
        Returns:
            - dataset: original PandasDataset
        """
        # For temporal data
        if dataset.temporal_data is not None and self.temporal_norm_parameters is not None:
            dataset.temporal_data = renormalization(dataset.temporal_data, self.temporal_norm_parameters)
        # For static data
        if dataset.static_data is not None and self.static_norm_parameters is not None:
            dataset.static_data = renormalization(dataset.static_data, self.static_norm_parameters)

        return dataset


class Normalizer(BaseEstimator, DataPreprocessorMixin):
    """Normalize the data.
    
    Attributes:
        - normalizer_name: 'minmax' or 'standard'    """

    def __init__(self, normalizer_type):
        self.normalizer_type = normalizer_type
        if self.normalizer_type == "minmax":
            self.norm = MinMaxNormalizer()
        elif self.normalizer_type == "standard":
            self.norm = StandardNormalizer()
        else:
            self.norm = None

    def fit(self, dataset):
        if self.norm is not None:
            self.norm.fit(dataset)

    def transform(self, dataset):
        if self.norm is not None:
            dataset = self.norm.transform(dataset)
        return dataset

    def fit_transform(self, dataset):
        """Transform original dataset to standard or minmax normalized dataset.
        
        Args:
            - dataset: original PandasDataset
        
        Returns:
            - dataset: normalized PandasDataset
            - norm_parameters: normalization parameters for renomalization
        """
        self.fit(dataset)
        return self.transform(dataset)


class ProblemMaker(BaseEstimator, DataPreprocessorMixin):
    """Define temporal, static, time, label, and treatment features.
    
    Attributes:
        - problem: 'online' or 'one-shot'
        - label: label names in list format
        - max_seq_len: maximum sequence length
        - treatment: the feature names for treatment features
        - window: set labels for window time ahead prediction
    """

    def __init__(self, problem, label, max_seq_len, treatment=None, window=0):
        assert problem in ["online", "one-shot"]
        self.problem = problem
        self.label = label
        self.max_seq_len = max_seq_len
        self.treatment = treatment
        self.window = window

    def pad_sequence(self, x):
        """Returns numpy array for predictor model training and testing after padding.

        Args:
            - x: temporal data in DataFrame

        Returns:
            - x_hat: preprocessed temporal data in 3d numpy array
        """
        uniq_id = np.unique(x["id"])
        x_hat = list()
        # For each patient
        for i in tqdm(range(len(uniq_id))):
            idx_x = x.index[x["id"] == uniq_id[i]]
            if len(idx_x) >= self.max_seq_len:
                temp_x = x.loc[idx_x[: self.max_seq_len]]
                temp_x = index_reset(temp_x)
            # Padding
            else:
                temp_x = padding(x.loc[idx_x], self.max_seq_len)

            x_hat = x_hat + [temp_x]
        return pd_list_to_np_array(x_hat, ["id"])

    def sliding_window_label(self, y):
        """Set sliding window label.
        
        Set labels for window ahead prediction.
        
        Args:
            - y: labels
        
        Returns:
            - y: sliding window label
        """
        if self.window > 0:
            y[:, : (self.max_seq_len - self.window), :] = y[:, self.window :, :]
            y[:, (self.max_seq_len - self.window) :, :] = -1
        return y

    def fit(self, dataset):
        pass

    def transform(self, dataset):
        return self.fit_transform(dataset)

    def fit_transform(self, dataset):
        """Transform the dataset based on the Pandas Dataframe to numpy array.
        
        Returned dataset has temporal, static, time, label and treatment features
        
        Args:
            - dataset: original dataset
            
        Returns:
            - dataset: defined dataset for the certain problem
        """
        x = dataset.temporal_data
        s = dataset.static_data

        # 1. Label define
        # For temporal labels
        if self.problem == "online":
            assert x is not None
            y = self.pad_sequence(x[["id"] + self.label])
            # if self.window > 0
            y = self.sliding_window_label(y)
            x = x.drop(self.label, axis=1)
        # For static labels
        else:
            assert s is not None
            y = np.asarray(s[self.label])
            s = s.drop(self.label, axis=1)

        # 2. Time define
        time = self.pad_sequence(x[["id", "time"]])
        x = x.drop(["time"], axis=1)

        # 3. Treatment define
        # TODO: mixing static and temporal treatment is not allowed
        if self.treatment is None:
            treatment = np.zeros([0])
        else:
            if self.treatment[0] in x.columns:
                treatment = self.pad_sequence(x[["id"] + self.treatment])
                x = x.drop(self.treatment, axis=1)
            elif self.treatment[0] in s.columns:
                treatment = np.asarray(s[self.treatment])
                s = s.drop(self.treatment, axis=1)
            else:
                raise ValueError("Treatment {} is not found in data set.".format(self.treatment[0]))

        # Set temporal and static features
        temporal_features = list_diff(x.columns.values.tolist(), ["id", "time"]) if x is not None else None
        static_features = list_diff(s.columns.values.tolist(), ["id"]) if s is not None else None

        # Feature name for visualization
        feature_name = {
            "temporal": temporal_features,
            "static": static_features,
            "treatment": self.treatment,
            "label": self.label,
        }

        # Set temporal features
        x = self.pad_sequence(x)
        # Set static features
        s = np.asarray(s.drop(columns=["id"]))

        # Define PandasDataset
        dataset.define_feature(x, s, y, treatment, time, feature_name, self.problem, self.label)

        return dataset
