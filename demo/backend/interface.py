from typing import Dict, List, NamedTuple, Optional
from datasets import PandasDataset
from preprocessing import ProblemMaker

import numpy as np


class DataSource(NamedTuple):
    data_name: str
    data_directory: str
    train_static_filename: str
    train_temporal_filename: str
    test_static_filename: str
    test_temporal_filename: str


class Results(NamedTuple):
    dataset_training: PandasDataset
    dataset_testing: PandasDataset
    test_y_hat: Optional[np.ndarray]
    test_ci_hat: Optional[np.ndarray]
    test_s_hat: Optional[np.ndarray]


# TODO: Re-evaluate how the ExtraSettings are passed:
class ExtraSettings(NamedTuple):
    model_name: str
    model_parameters: Dict
    metric_name: str
    metric_parameters: Dict
    task: str
    projection_horizon: Optional[int] = None


class DataBundle(NamedTuple):
    temporal_feature_names: List
    testing_id_row_map: Dict
    raw_dataset_training: PandasDataset
    raw_dataset_testing: PandasDataset


class ProblemBundle(NamedTuple):
    problem_maker: ProblemMaker
    extra_settings: ExtraSettings
    results: Results
