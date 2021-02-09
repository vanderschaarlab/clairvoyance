import os
from typing import Tuple, Union, Dict
from datasets import CSVLoader, PandasDataset

import numpy as np
import pandas as pd

from .interface import DataSource, DataBundle


# TODO: Deal with this demo data stuff properly.
STATIC_DATA_LOAD_LIMIT = 10_000
TEMPORAL_DATA_LOAD_LIMIT = 100_000


def none_to_empty_list(val):
    return [] if val is None else val


def empty_list_to_none(val):
    return None if val == [] else val


def _get_temporal_feature_names(raw_dataset_training):
    columns = list(raw_dataset_training.temporal_data.columns)
    assert (columns[0], columns[1]) == ("id", "time")
    return columns[2:]


def _get_testing_id_row_map(raw_dataset_testing):

    # Unless the below assertion is satisfied, the id-row map will not be correct.
    static_ids = list(raw_dataset_testing.static_data["id"])
    temporal_ids = list(raw_dataset_testing.temporal_data["id"].unique())
    assert static_ids == temporal_ids

    return dict(zip(static_ids, range(0, len(static_ids))))


# TODO: The content of this function requires further work.
def postprocess_raw_data(raw_dataset_training, raw_dataset_testing) -> DataBundle:
    return DataBundle(
        temporal_feature_names=_get_temporal_feature_names(raw_dataset_training),
        testing_id_row_map=_get_testing_id_row_map(raw_dataset_testing),
        raw_dataset_training=raw_dataset_training,
        raw_dataset_testing=raw_dataset_testing,
    )


def _prune_by_loaded_data(static_data: pd.DataFrame, temporal_data: pd.DataFrame):
    """Relevant to when using `static_only_nrows`/`temporal_only_nrows` in `data_loader.load()`.
    If there are fewer unique `id`s in the temporal loaded data than there are `id`s in static data, 
    prune the static data appropriately, do the reverse otherwise (i.e. keep only the intersection of `id`s).
    """
    static_ids = static_data["id"].unique()
    temporal_ids = temporal_data["id"].unique()
    intersection = np.intersect1d(static_ids, temporal_ids)
    if len(intersection) < len(static_ids):
        static_data = static_data.loc[static_data["id"].isin(intersection), :]
    if len(intersection) < len(temporal_ids):
        temporal_data = temporal_data.loc[temporal_data["id"].isin(intersection), :]
    return static_data, temporal_data


loaded_data_sources: Dict = dict()


def load_data(data: Union[DataSource, str]) -> Tuple[PandasDataset, PandasDataset]:
    global loaded_data_sources  # pylint: disable=global-statement

    if isinstance(data, DataSource):
        key = data.data_name
        data_source: DataSource = data
    else:
        key = data
        if key not in loaded_data_sources:
            raise ValueError(f"DataSource named '{key}' hasn't been defined.")

    if key not in loaded_data_sources:
        loaded_data_sources[key] = dict()

        data_loader_training = CSVLoader(
            static_file=os.path.join(data_source.data_directory, data_source.train_static_filename),
            temporal_file=os.path.join(data_source.data_directory, data_source.train_temporal_filename),
        )
        data_loader_testing = CSVLoader(
            static_file=os.path.join(data_source.data_directory, data_source.test_static_filename),
            temporal_file=os.path.join(data_source.data_directory, data_source.test_temporal_filename),
        )

        dataset_training = data_loader_training.load(
            static_only_nrows=STATIC_DATA_LOAD_LIMIT, temporal_only_nrows=TEMPORAL_DATA_LOAD_LIMIT
        )
        dataset_testing = data_loader_testing.load(
            static_only_nrows=STATIC_DATA_LOAD_LIMIT, temporal_only_nrows=TEMPORAL_DATA_LOAD_LIMIT
        )

        if STATIC_DATA_LOAD_LIMIT is not None or TEMPORAL_DATA_LOAD_LIMIT is not None:
            dataset_training.static_data, dataset_training.temporal_data = _prune_by_loaded_data(
                dataset_training.static_data, dataset_training.temporal_data
            )
            dataset_testing.static_data, dataset_testing.temporal_data = _prune_by_loaded_data(
                dataset_testing.static_data, dataset_testing.temporal_data
            )

        loaded_data_sources[key]["dataset_training"] = dataset_training
        loaded_data_sources[key]["dataset_testing"] = dataset_testing

    return loaded_data_sources[key]["dataset_training"], loaded_data_sources[key]["dataset_testing"]
