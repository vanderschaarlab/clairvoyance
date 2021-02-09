"""Static and temporal imputation.

(1) Static imputation (6 options)
- BasicImputation: mean, median
- StandardImputation: mice, missforest, knn
- NNImputation: gain

(2) Temporal imputation (8 options)
- BasicImputation: mean, median
- Interpolation: linear, quadratic, cubic, spline
- NNImputation: tgain, mrnn
"""

# Necessary packages
import numpy as np
from base import BaseEstimator, DataPreprocessorMixin

# Static imputation
from missingpy import MissForest, KNNImputer

# TODO: Resolve the below:
# from fancyimpute import IterativeImputer
from imputation.static.gain_imputation import GainImputation

# Temporal imputation
from imputation.temporal.interpolation import interpolation
from imputation.temporal.tgain_imputation import TGainImputation
from imputation.temporal.mrnn_imputation import MRnnImputation

# Utils
from imputation.imputation_utils import rounding


class BasicImputation(BaseEstimator, DataPreprocessorMixin):
    """Mean and median imputations.
    
    Attributes:
        - imputation_model_name: 'mean' or 'median'
        - data_type: 'temporal' or 'static'
    """

    def __init__(self, imputation_model_name, data_type):
        # Only allow for certain options
        assert imputation_model_name in ["mean", "median"]
        assert data_type in ["temporal", "static"]

        self.imputation_model_name = imputation_model_name
        self.data_type = data_type
        # Functions for computing mean and median without nan values
        self.agg_func = np.nanmean if imputation_model_name == "mean" else np.nanmedian
        # Save mean or median of the data
        self.vals = None

    def mean_median_computation(self, data):
        """Compute mean or median without nan and padded values.
        
        Args:
            - data: incomplete data
            
        Returns:
            - vals: mean or median of each variable
        """
        new_data = data.copy()
        # For temporal features, reshape for 2d array
        if len(data.shape) == 3:
            new_data = np.reshape(new_data, [data.shape[0] * data.shape[1], data.shape[2]])
        # Excluding padded values
        idx = np.where(new_data[:, 0] != -1)[0]
        # Compute mean or median of the data
        vals = self.agg_func(new_data[idx, :], axis=0)

        return vals

    def fit(self, dataset):
        """Compute mean or median values.
        
        Args:
            - dataset: incomplete dataset
        """
        if self.data_type == "static":
            if dataset.static_feature is not None:
                self.vals = self.mean_median_computation(dataset.static_feature)
        elif self.data_type == "temporal":
            if dataset.temporal_feature is not None:
                self.vals = self.mean_median_computation(dataset.temporal_feature)

        return

    def transform(self, dataset):
        """Return mean or median imputed dataset.
        
        Args:
            - dataset: incomplete dataset
        
        Returns:
            - dataset: mean or median imputed dataset
        """
        # Only when fitted
        assert self.vals is not None

        if self.data_type == "static":
            if dataset.static_feature is not None:
                nan_mask = np.isnan(dataset.static_feature).astype(np.float)
                dataset.static_feature = (
                    np.nan_to_num(dataset.static_feature, 0) * (1 - nan_mask) + self.vals * nan_mask
                )
        elif self.data_type == "temporal":
            if dataset.temporal_feature is not None:
                nan_mask = np.isnan(dataset.temporal_feature).astype(np.float)
                dataset.temporal_feature = (
                    np.nan_to_num(dataset.temporal_feature, 0) * (1 - nan_mask) + self.vals * nan_mask
                )

        return dataset

    def fit_transform(self, dataset):
        """Fit and transform. Return imputed data.
        
        Args:
            - dataset: incomplete dataset
        """
        self.fit(dataset)
        return self.transform(dataset)


class Interpolation(BaseEstimator, DataPreprocessorMixin):
    """Temporal data interpolation.
    
    Attributes:
        - interpolation_model_name: 'linear', 'quadratic', 'cubic', 'spline'
        - data_type: 'temporal'
    """

    def __init__(self, interpolation_model_name, data_type):
        # Only allow for certain options
        assert interpolation_model_name in ["linear", "quadratic", "cubic", "spline"]
        assert data_type == "temporal"

        self.interpolation_model_name = interpolation_model_name
        # Do median imputation when the entire sequence is missing
        self.median_imputation = None

    def fit(self, dataset):
        """Compute median values for median imputation.
        
        Interpolation does not need fitting (before). But median imputation needs fitting.
        
        Args:
            - dataset: incomplete dataset
        """
        if dataset.temporal_feature is not None:
            # Compute median values via Median imputation
            self.median_imputation = BasicImputation(imputation_model_name="median", data_type="temporal")
            self.median_imputation.fit(dataset)

        return

    def transform(self, dataset):
        """Return interpolated dataset & median imputed dataset.
        
        Args:
            - dataset: incomplete dataset
        
        Returns:
            - dataset: interpolated dataset
        """
        # Only after fitting
        assert self.median_imputation is not None

        if dataset.temporal_feature is not None:
            # Interpolate temporal data if at least one value is observed
            dataset.temporal_feature = interpolation(
                dataset.temporal_feature, dataset.time, self.interpolation_model_name
            )
            # Do median imputation for the sequence without any observed data
            dataset = self.median_imputation.transform(dataset)

        return dataset

    def fit_transform(self, dataset):
        """Fit and transform. Return imputed data
        
        Args:
            - dataset: incomplete dataset
        """
        self.fit(dataset)
        return self.transform(dataset)


class NNImputation(BaseEstimator, DataPreprocessorMixin):
    """Neural network based imputation method.
    
    Attributes:
        - imputation_model_name: 'tgain' or 'mrnn' for temporal data, 'gain' for static data
        - data_type: 'static' or 'temporal'
    """

    def __init__(self, imputation_model_name, data_type):
        # Only allow for certain options
        assert data_type in ["static", "temporal"]
        assert imputation_model_name in ["tgain", "mrnn", "gain"]

        self.imputation_model_name = imputation_model_name
        self.data_type = data_type
        # Initialize the modules
        self.nn_temporal_imputation_model = None
        self.nn_static_imputation_model = None

    def fit(self, dataset):
        """Train NN based imputation modules
        
        Args:
            - dataset: incomplete dataset
        """
        if self.data_type == "temporal":
            if dataset.temporal_feature is not None:
                # Define temporal imputation module
                if self.imputation_model_name == "tgain":
                    self.nn_temporal_imputation_model = TGainImputation(file_name="tgain")
                elif self.imputation_model_name == "mrnn":
                    self.nn_temporal_imputation_model = MRnnImputation(file_name="mrnn")
                # Train temporal imputation module
                self.nn_temporal_imputation_model.fit(dataset.temporal_feature)
        elif self.data_type == "static":
            if dataset.static_feature is not None:
                # Define static imputation module
                if self.imputation_model_name == "gain":
                    self.nn_static_imputation_model = GainImputation(file_name="gain")
                # Train static imputation module
                self.nn_static_imputation_model.fit(dataset.static_feature)

        return dataset

    def transform(self, dataset):
        """Return imputed data using NN imputation modules.
        
        Args:
            - dataset: incomplete dataset
        
        Returns:
            - dataset: imputed dataset after NN based imputation
        """
        if self.data_type == "temporal":
            if dataset.temporal_feature is not None:
                if self.imputation_model_name in ["tgain", "mrnn"]:
                    assert self.nn_temporal_imputation_model is not None
                    # Temporal data imputation
                    dataset.temporal_feature = self.nn_temporal_imputation_model.transform(dataset.temporal_feature)
        elif self.data_type == "static":
            if dataset.static_feature is not None:
                if self.imputation_model_name in ["gain"]:
                    assert self.nn_static_imputation_model is not None
                    # Static data imputation
                    dataset.static_feature = self.nn_static_imputation_model.transform(dataset.static_feature)

        return dataset

    def fit_transform(self, dataset):
        """Fit and transform. Return imputed data
        
        Args:
            - dataset: incomplete dataset
        """
        self.fit(dataset)
        return self.transform(dataset)


class StandardImputation(BaseEstimator, DataPreprocessorMixin):
    """Standard imputation method for static data.
        
    Reference 1: https://pypi.org/project/missingpy/
    Reference 2: https://s3.amazonaws.com/assets.datacamp.com/production/course_17404/slides/chapter4.pdf
    
    Attributes:
        - imputation_model_name: 'mice', 'missforest', 'knn'
        - data_type: 'static'
    """

    def __init__(self, imputation_model_name, data_type):
        # Only allow for certain options
        assert data_type == "static"
        assert imputation_model_name in ["mice", "missforest", "knn"]

        self.imputation_model_name = imputation_model_name
        self.data_type = data_type
        # Initialize the imputation model
        self.imputation_model = None

    def fit(self, dataset):
        """Train standard imputation model.
        
        Args:
            - dataset: incomplete dataset
        """
        if dataset.static_feature is not None:
            # MICE
            if self.imputation_model_name == "mice":
                # TODO: Resolve the below:
                raise NotImplementedError("IterativeImputer not implemented due to versioning issues with fancyimpute")
                # self.imputation_model = IterativeImputer()
            # MissForest
            elif self.imputation_model_name == "missforest":
                self.imputation_model = MissForest()
            # KNN
            elif self.imputation_model_name == "knn":
                self.imputation_model = KNNImputer()

            self.imputation_model.fit(dataset.static_feature)

        return

    def transform(self, dataset):
        """Return imputed dataset by standard imputation.
        
        Args:
            - dataset: incomplete dataset
        
        Returns:
            - dataset: imputed dataset by standard imputation.
        """
        assert self.imputation_model is not None

        if dataset.static_feature is not None:
            # Standard imputation
            data_imputed = self.imputation_model.transform(dataset.static_feature)
            # Rounding
            dataset.static_feature = rounding(dataset.static_feature, data_imputed)

        return dataset

    def fit_transform(self, dataset):
        """Fit and transform. Return imputed data
        
        Args:
            - dataset: incomplete dataset
        """
        self.fit(dataset)
        return self.transform(dataset)


class Imputation(BaseEstimator, DataPreprocessorMixin):
    """Missing data imputation.
    
    Attributes:
        - imputation_model_name: 6 possible static imputations and 8 possible temporal imputations
        - data_type: 'temporal' or 'static'
    """

    def __init__(self, imputation_model_name, data_type):
        # Only allow for certain options
        assert imputation_model_name in [
            "mean",
            "median",
            "mice",
            "missforest",
            "knn",
            "gain",
            "linear",
            "quadratic",
            "cubic",
            "spline",
            "mrnn",
            "tgain",
        ]
        assert data_type in ["temporal", "static"]

        self.imputation_model_name = imputation_model_name
        self.data_type = data_type
        self.imputation_model = None

        # (1) Static imputation: 6 options
        if self.data_type == "static":
            if imputation_model_name in ["mean", "median"]:
                self.imputation_model = BasicImputation(imputation_model_name=imputation_model_name, data_type="static")
            elif imputation_model_name in ["mice", "missforest", "knn"]:
                self.imputation_model = StandardImputation(
                    imputation_model_name=imputation_model_name, data_type="static"
                )
            elif imputation_model_name in ["gain"]:
                self.imputation_model = NNImputation(imputation_model_name=imputation_model_name, data_type="static")

        # (2) Temporal imputation: 8 options
        elif self.data_type == "temporal":
            if imputation_model_name in ["mean", "median"]:
                self.imputation_model = BasicImputation(
                    imputation_model_name=imputation_model_name, data_type="temporal"
                )
            elif imputation_model_name in ["linear", "quadratic", "cubic", "spline"]:
                self.imputation_model = Interpolation(
                    interpolation_model_name=imputation_model_name, data_type="temporal"
                )
            elif imputation_model_name in ["mrnn", "tgain"]:
                self.imputation_model = NNImputation(imputation_model_name=imputation_model_name, data_type="temporal")

    def fit(self, dataset):
        """Train imputation model.
        
        Args:
            - dataset: incomplete dataset
        """
        self.imputation_model.fit(dataset)
        return

    def transform(self, dataset):
        """Return imputed dataset by standard imputation.
        
        Args:
            - dataset: incomplete dataset
        
        Returns:
            - dataset: imputed dataset by standard imputation.
        """
        assert self.imputation_model is not None
        dataset = self.imputation_model.transform(dataset)
        return dataset

    def fit_transform(self, dataset):
        """Fit and transform. Return imputed data
        
        Args:
            - dataset: incomplete dataset
        """
        self.fit(dataset)
        return self.transform(dataset)
