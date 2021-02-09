"""Base classes for all estimators.
"""

# Necessary packages
from collections import defaultdict
import inspect
from utils import concate_xs, concate_xt


class BaseEstimator:
    """Base class for all estimators.
    
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values() if p.name != "self" and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "Estimators should always specify their parameters in the signature of their __init__."
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
                If True, will return the parameters for this estimator and
                contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
                Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
            except AttributeError:
                value = None
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.
        
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        
        Parameters
        ----------
        **params : dict
                Estimator parameters.
        Returns
        -------
        self : object
                Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError(
                    "Invalid parameter %s for estimator %s. "
                    "Check the list of available parameters "
                    "with `estimator.get_params().keys()`." % (key, self)
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def __getstate__(self):
        try:
            state = super().__getstate__()
        except AttributeError:
            state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        try:
            super().__setstate__(state)
        except AttributeError:
            self.__dict__.update(state)


class DataLoaderMixin:
    """Mixin class for all data loaders."""

    def load(self):
        raise NotImplementedError


class TransformerMixin:
    """Mixin class for all transformers in scikit-learn."""

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.
        
        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
                Training set.
        y : ndarray of shape (n_samples,), default=None
                Target values.
        **fit_params : dict
                Additional fit parameters.
        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
                Transformed array.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)


class DataPreprocessorMixin:
    """Mixin class for all pre-processing data filters."""

    def fit_transform(self, dataset):
        """Perform fit on df and returns a new X with invalid values filtered.
        
        Parameters
        ----------
        dataset : a data set
                Input data.
        Returns
        -------
        new dataset: Transformed data.
        """
        raise NotImplementedError


class PredictorMixin:
    def __init__(self, static_mode=None, time_mode=None, task=None):
        self.static_mode = static_mode
        self.time_mode = time_mode
        self.task = task

    """Mixin class for all predictors."""

    def fit(self, dataset):
        """Perform fit on df.
        
        Parameters
        ----------
        dataset : a data set
                Input data.
        Returns
        -------
        None
        """
        raise NotImplementedError

    def predict(self, dataset):
        """Perform predict on df.
        
        Parameters
        ----------
        dataset : a data set
                Input data.
        Returns
        -------
        None
        """
        raise NotImplementedError

    def save_model(self, model_path):
        raise NotImplementedError

    def load_model(self, model_path):
        raise NotImplementedError

    @staticmethod
    def get_hyperparameter_space():
        raise NotImplementedError

    def new(self, model_id):
        raise NotImplementedError

    def _data_preprocess(self, dataset, fold, split):
        """Preprocess the dataset.

        Returns feature and label.

        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Cross validation fold
            - split: 'train', 'valid' or 'test'

        Returns:
            - x: temporal feature
            - y: labels
        """
        # Set temporal, static, label, and time information
        x, s, y, t, _ = dataset.get_fold(fold, split)

        if self.static_mode == "concatenate":
            if s is not None:
                x = concate_xs(x, s)

        if self.time_mode == "concatenate":
            if t is not None:
                x = concate_xt(x, t)

        return x, y
