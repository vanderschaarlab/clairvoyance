import os
from typing import Callable, Dict
import pickle

import numpy as np

from datasets import PandasDataset
from preprocessing import ProblemMaker

from .interface import DataSource, Results, ExtraSettings, ProblemBundle
from .utils import load_data, none_to_empty_list, empty_list_to_none, postprocess_raw_data
from .pipelines import timeseries_prediction_pl, active_sensing_pl, ite_pl


class Problem(object):
    def __init__(
        self,
        data_source: DataSource,
        problem_maker: ProblemMaker,
        extra_settings: ExtraSettings,
        pipeline: Callable[[ProblemMaker, ExtraSettings, PandasDataset, PandasDataset], Results],
        results_filepath: str,
    ) -> None:
        self.data_source = data_source
        self.problem_maker = problem_maker
        self.extra_settings = extra_settings
        self.pipeline = pipeline
        self.results_filepath = results_filepath
        self.raw_dataset_training, self.raw_dataset_testing = load_data(self.data_source)

    def _results_ready(self) -> bool:
        if os.path.exists(self.results_filepath):
            return True
        else:
            return False

    def execute(self) -> Results:

        if not self._results_ready():
            results = self.pipeline(
                self.problem_maker, self.extra_settings, self.raw_dataset_training, self.raw_dataset_testing
            )
            self.save_results(results)

        return self.load_results()

    def save_results(self, results: Results) -> None:
        np.savez(
            self.results_filepath,
            test_y_hat=none_to_empty_list(results.test_y_hat),
            test_ci_hat=none_to_empty_list(results.test_ci_hat),
            test_s_hat=none_to_empty_list(results.test_s_hat),
        )
        pickle_path = self.results_filepath.replace(".npz", ".p")
        pickled = {"dataset_training": results.dataset_training, "dataset_testing": results.dataset_testing}
        pickle.dump(pickled, open(pickle_path, "wb"))

    def load_results(self) -> Results:
        npz = np.load(self.results_filepath)
        pickle_path = self.results_filepath.replace(".npz", ".p")
        pickled = pickle.load(open(pickle_path, "rb"))
        return Results(
            dataset_training=pickled["dataset_training"],
            dataset_testing=pickled["dataset_testing"],
            test_y_hat=empty_list_to_none(npz["test_y_hat"]),
            test_ci_hat=empty_list_to_none(npz["test_ci_hat"]),
            test_s_hat=empty_list_to_none(npz["test_s_hat"]),
        )


def _populate_backend_contents(problem_defs: Dict) -> Dict:

    backend_contents: Dict = dict()

    for data_source_name, data_source_problem_defs in problem_defs.items():
        print(f"- Preparing data source: '{data_source_name}'...")

        if data_source_problem_defs is not None:
            backend_contents[data_source_name] = {"data": None, "problems": dict()}

            # Populate dataset info.
            dataset_training, dataset_testing = load_data(data_source_name)
            backend_contents[data_source_name]["data"] = postprocess_raw_data(dataset_training, dataset_testing)

            # Execute / load problems.
            for problem_setting_name, problem in data_source_problem_defs.items():
                print(f"\t- Preparing problem setting: '{data_source_name}'/'{problem_setting_name}'...")

                if problem is not None:

                    results = problem.execute()

                    backend_contents[data_source_name]["problems"][problem_setting_name] = ProblemBundle(
                        problem_maker=problem.problem_maker, extra_settings=problem.extra_settings, results=results,
                    )

                else:
                    print(f"\t  > (Problem setting '{data_source_name}'/'{problem_setting_name}' not defined)")
                    backend_contents[data_source_name]["problems"][problem_setting_name] = None

        else:
            print(f"  > (Data source '{data_source_name}' has no defined problem settings)")
            backend_contents[data_source_name] = None

    return backend_contents


def load_backend(working_dir: str) -> Dict:

    # TODO: These var names.
    # Definitions.
    mimic = DataSource(
        data_name="mimic",
        data_directory=os.path.abspath("../datasets/data/mimic"),
        train_static_filename="mimic_static_train_data.csv.gz",
        train_temporal_filename="mimic_temporal_train_data_eav.csv.gz",
        test_static_filename="mimic_static_test_data.csv.gz",
        test_temporal_filename="mimic_temporal_test_data_eav.csv.gz",
    )
    mimic_online_ventilator = ProblemMaker(
        problem="online", label=["ventilator"], max_seq_len=24, treatment=None, window=4,
    )
    mimic_oneshot_death = ProblemMaker(problem="one-shot", label=["death"], max_seq_len=24, treatment=None, window=0,)
    mimic_online_treatment = ProblemMaker(
        problem="online", label=["death"], max_seq_len=24, treatment=["ventilator"], window=1,
    )

    problem_defs = {
        "mimic": {
            # ---
            "time_series_prediction": Problem(
                data_source=mimic,
                problem_maker=mimic_online_ventilator,
                extra_settings=ExtraSettings(
                    model_name="gru",
                    model_parameters={
                        "h_dim": 100,
                        "n_layer": 2,
                        "n_head": 2,
                        "batch_size": 128,
                        "epoch": 20,
                        "model_type": "gru",
                        "learning_rate": 0.001,
                        "static_mode": "Concatenate",
                        "time_mode": "Concatenate",
                        "verbose": True,
                    },
                    metric_name="auc",
                    task="classification",
                    metric_parameters={
                        "problem": mimic_online_ventilator.problem,
                        "label_name": mimic_online_ventilator.label,
                    },
                ),
                pipeline=timeseries_prediction_pl,
                results_filepath=os.path.join(working_dir, "data_mimic_main_ts.npz"),
            ),
            # ---
            "active_sensing": Problem(
                data_source=mimic,
                problem_maker=mimic_oneshot_death,
                extra_settings=ExtraSettings(
                    model_name="asac",
                    model_parameters={
                        "h_dim": 100,
                        "n_layer": 2,
                        "batch_size": 128,
                        "epoch": 20,
                        "model_type": "gru",
                        "learning_rate": 0.001,
                        "static_mode": "Concatenate",
                        "time_mode": "Concatenate",
                        "verbose": True,
                    },
                    metric_name="auc",
                    task="classification",
                    metric_parameters={"problem": mimic_oneshot_death.problem, "label_name": mimic_oneshot_death.label},
                ),
                pipeline=active_sensing_pl,
                results_filepath=os.path.join(working_dir, "data_mimic_main_as.npz"),
            ),
            # ---
            # "ITE": Problem(
            #     data_source=mimic,
            #     problem_maker=mimic_online_treatment,
            #     extra_settings=ExtraSettings(
            #         model_name="CRN",
            #         model_parameters={
            #             "encoder_rnn_hidden_units": 128,
            #             "encoder_br_size": 64,
            #             "encoder_fc_hidden_units": 128,
            #             "encoder_learning_rate": 0.001,
            #             "encoder_batch_size": 256,
            #             "encoder_keep_prob": 0.9,
            #             "encoder_num_epochs": 100,
            #             "encoder_max_alpha": 1.0,
            #             "decoder_br_size": 64,
            #             "decoder_fc_hidden_units": 128,
            #             "decoder_learning_rate": 0.001,
            #             "decoder_batch_size": 512,
            #             "decoder_keep_prob": 0.9,
            #             "decoder_num_epochs": 100,
            #             "decoder_max_alpha": 1.0,
            #             "projection_horizon": 5,
            #             "static_mode": "concatenate",
            #             "time_mode": "concatenate",
            #         },
            #         metric_name="auc",
            #         task="classification",
            #         metric_parameters={"problem": mimic_online_treatment.problem, "label_name": mimic_online_treatment.label},
            #         projection_horizon=5,
            #     ),
            #     pipeline=ite_pl,
            #     results_filepath=os.path.join(working_dir, "data_mimic_main_ite.npz"),
            # ),
        },
        # TODO: Other datasets.
    }

    # Prepare backend.
    print("Loading Clairvoyance demo backend...")
    backend_contents = _populate_backend_contents(problem_defs)
    print("Loading Clairvoyance demo backend DONE.")

    return backend_contents
