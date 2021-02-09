"""Replace outlier values to NaN.

(1) FilterOutOfRange: Replace outlier values to NaN by manually defined range
- remove_out_of_range: set any values out of range to NaN
(2) FilterNegative: Replace the negative values to NaN
- remove_negative: set any negative values to NaN
(3) FilterOutOfConfidence: Replace the values to NaN by confidence
- confidence_interval: compute the range of confidence interval
"""

# Necessary packages
import numpy as np
from base import BaseEstimator, DataPreprocessorMixin
from datasets import PandasDataset


def remove_out_of_range(df, range_list):
    """Set any values out of range to NAN.
    
    Args:
        - df: a wide format df (e.g. the result of data_loader)
        - range_list: a list of dictionaries. Each dictionary has three keys 'variable', 'high' and 'low'.
                                    e.g. {'variable': 'aao2', 'high': 700., 'low':10.}
    Returns: 
        - df: dataset with invalid values set to NAN
    """
    if range_list is None:
        return df
    for f in range_list:
        if f["variable"] in df.columns:
            col = df[f["variable"]]
            df[f["variable"]] = col.mask((col < f["low"]) | (col > f["high"]))
    return df


class FilterOutOfRange(BaseEstimator, DataPreprocessorMixin):
    """Set any values out of range to NAN.

    Attributes:
        - range_list: a list of dictionaries. Each dictionary has three keys 'variable', 'high' and 'low'.
                                    e.g. {'variable': 'aao2', 'high': 700., 'low':10.}
    """

    def __init__(self, range_list=None):
        self.range_list = None
        if range_list is not None:
            for f in range_list:
                assert f["variable"] is not None
                assert f["low"] is not None
                assert f["high"] is not None
            self.range_list = range_list

    def fit(self, dataset):
        pass

    def transform(self, dataset):
        return self.fit_transform(dataset)

    def fit_transform(self, dataset):
        """Replace the values outside of the range to NaN.
        
        Args:
            - dataset: original dataset
            
        Returns:
            - dataset: dataset without outlier
        """
        if dataset.static_data is not None:
            static_data = remove_out_of_range(dataset.static_data, self.range_list)
        else:
            static_data = None

        if dataset.temporal_data is not None:
            temporal_data = remove_out_of_range(dataset.temporal_data, self.range_list)
        else:
            temporal_data = None
        return PandasDataset(static_data=static_data, temporal_data=temporal_data)


def remove_negative_value(df):
    """Replace negative values to NaN.
    
    Args:
        - df: original data.
    
    Returns:
        - df: data without negative value.
    """
    num = df._get_numeric_data()
    num[num < 0] = np.nan
    return df


class FilterNegative(BaseEstimator, DataPreprocessorMixin):
    """Replace the negative values to NaN.
    """

    def fit(self, dataset):
        pass

    def transform(self, dataset):
        return self.fit_transform(dataset)

    def fit_transform(self, dataset):
        """Replace the negative values to NaN.
        
        Args:
            - dataset: original dataset
            
        Returns:
            - dataset: dataset without negative values
        """
        if dataset.static_data is not None:
            static_data = remove_negative_value(dataset.static_data)
        else:
            static_data = None
        if dataset.temporal_data is not None:
            temporal_data = remove_negative_value(dataset.temporal_data)
        else:
            temporal_data = None
        return PandasDataset(static_data=static_data, temporal_data=temporal_data)


def confidence_interval(df, confidence_range):
    """Compute confidence interval of each variable.
    
    Args:
        - df: original data.
        - confidence_range: '90', '95', '99'
        
    Returns:
        - output_range: computed confidence intervals
    """
    output_range = list()
    confidence_dict = {"90": 1.645, "95": 1.96, "99": 2.576}
    confidence_const = confidence_dict[str(confidence_range)]

    for feature in df.columns:
        temp_range = dict()
        temp_range["variable"] = feature
        temp_range["low"] = df[feature].mean() - confidence_const * df[feature].std()
        temp_range["high"] = df[feature].mean() + confidence_const * df[feature].std()
        output_range = output_range + [temp_range]

    return output_range


class FilterOutOfConfidence(BaseEstimator, DataPreprocessorMixin):
    """Replace the values to NaN by confidence.
    
    Attributes:
        - confidence_range: '90', '95', '99'
    """

    def __init__(self, confidence_range):
        self.confidence_range = None
        self.static_range = None
        self.temporal_range = None
        if confidence_range is not None:
            assert confidence_range in [90, 95, 99]
            self.confidence_range = confidence_range

    def fit(self, dataset):
        """Compute confidence intervals.
        
        Args:
            - dataset: raw data
        """
        if dataset.static_data is not None:
            self.static_range = confidence_interval(dataset.static_data, self.confidence_range)

        if dataset.temporal_data is not None:
            self.temporal_range = confidence_interval(dataset.temporal_data, self.confidence_range)

    def transform(self, dataset):
        """Replace the values outside of the confidence interval to NaN.
        
        Args:
            - dataset: raw data
            
        Returns:
            - dataset: dataset without values outside of the confidence interval
        """
        if dataset.static_data is not None:
            static_data = remove_out_of_range(dataset.static_data, self.static_range)
        else:
            static_data = None

        if dataset.temporal_data is not None:
            temporal_data = remove_out_of_range(dataset.temporal_data, self.temporal_range)
        else:
            temporal_data = None

        return PandasDataset(static_data=static_data, temporal_data=temporal_data)

    def fit_transform(self, dataset):
        """Replace the values outside of the confidence interval to NaN.
        
        Args:
            - dataset: original dataset
            
        Returns:
            - dataset: dataset without values outside of the confidence interval
        """
        self.fit(dataset)
        return self.transform(dataset)
