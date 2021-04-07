"""Load datasets from raw files.

- Missing data is represented as blank (or NaN)
- Consist of two components:
    (1) temporal data
    (2) static data
    
- Following columns are necessary: id, time
    (1) id: patient id to connect between temporal data and static data
    (2) time: measurement time (only in the temporal data)
"""

# Necessary packages
import pandas as pd
from base import BaseEstimator, DataLoaderMixin
from .dataset import PandasDataset


class CSVLoader(BaseEstimator, DataLoaderMixin):
    """Load datasets from csv files.
    
    Attributes:
        - static_file: file name of static data
        - temporal_file: file name of temporal data
    """

    def __init__(self, static_file=None, temporal_file=None):
        self.static_file = static_file
        self.temporal_file = temporal_file

    def load(self, static_only_nrows=None, temporal_only_nrows=None):
        """Return both temporal and static datasets in PandasDataset format.
        """
        s = self._load_static(only_nrows=static_only_nrows)
        x = self._load_temporal(only_nrows=temporal_only_nrows)

        # -----
        s["id"] = s["id"].astype(int)
        s.sort_values(by="id", inplace=True)
        x.sort_values(by="id", inplace=True)
        # -----

        return PandasDataset(s, x)

    def _load_static(self, only_nrows):
        """Load static data from csv file (static_file).
        """
        if self.static_file is not None:
            try:
                static_data = pd.read_csv(self.static_file, delimiter=",", nrows=only_nrows)
                return static_data
            except:
                raise IOError("Static file (" + self.static_file + ") is not exist.")

        else:
            return None

    def _load_temporal(self, only_nrows):
        """Load temporal data from csv file (temporal file).
        Convert EAV format to WIDE format.
        """
        if self.temporal_file is not None:
            try:
                temporal_data = pd.read_csv(self.temporal_file, delimiter=",", nrows=only_nrows)
            except:
                raise IOError("Temporal file (" + self.temporal_file + ") is not exist.")
            # Convert EAV format to WIDE format
            return eav_to_wide(temporal_data)
        else:
            return None


def eav_to_wide(df):
    """Transform EAV format to WIDE format.
    
    Args:
        - df: EAV format dataframe
        
    Returns:
        - df_wide: WIDE format dataframe.    
    """
    # Original data needs the following four column name in order.
    col_names = list(df.columns)
    assert col_names[0] == "id"
    assert col_names[1] == "time"
    assert col_names[2] == "variable"
    assert col_names[3] == "value"

    # Convert EAV format to WIDE format
    df_wide = pd.pivot_table(df, index=["id", "time"], columns="variable", values="value").reset_index(level=[0, 1])
    return df_wide
