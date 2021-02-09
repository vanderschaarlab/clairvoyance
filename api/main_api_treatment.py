"""Main function for individual treatment effects estimation.

Pipeline
Step 1: Load dataset
    - data_name: mimic, ward, cf, mimic_antibiotics

Step 2: Preprocess dataset
    (0) NegativeFilter: Replace negative values to NaN
    (1) OneHotEncoder: One hot encoding certain features
    (2) Normalization (3 options): MinMax, Standard, None

Step 3: Define problem
    - problem: online
    - label_name: the column name for the label(s)
    - max_seq_len: maximum sequence length after padding
    - treatment: the column name for treatments

Step 4: Impute dataset
    (0) Static imputation (6 options): mean, median, mice, missforest, knn, gain
    (1) Temporal imputation (8 options): mean, median, linear, quadratic, cubic, spline, mrnn, tgain

Step 5: Feature selection
    - feature selection method (5 options): greedy-addtion, greedy-deletion, recursive-addition, recursive-deletion, None

Step 6: Treatment effect model fit and predict
    - predictive models (3 options): CRN (Counterfactual Recurrent Network), RMSN (Recurrent Marginal Structural Networks),
                                                                     GANITE

Step 7: Visualize results
    - metric_name (4 options): auc, apr, mse, mae
    (1) Visualize the performance on estimating factual outcomes
    (2) Visualize the counterfactual trajectories

"""

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")

import sys

sys.path.append("../")
from datasets import CSVLoader
from preprocessing import FilterNegative, OneHotEncoder, Normalizer, ProblemMaker
from imputation import Imputation
from feature_selection import FeatureSelection
from treatments import treatment_effects_model
from evaluation import Metrics
from evaluation import print_performance, print_counterfactual_predictions
from utils import PipelineComposer

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main(args):
    """Main function for individual treatment effect estimation.

    Args:
        - data loading parameters:
            - data_names: mimic, ward, cf, mimic_antibiotics

        - preprocess parameters:
            - normalization: minmax, standard, None
            - one_hot_encoding: input features that need to be one-hot encoded
            - problem: 'online'
                - 'online': prediction at every time stamps of the time-series
            - max_seq_len: maximum sequence length after padding
            - label_name: the column name for the label(s)
            - treatment: the column name for treatments

        - imputation parameters:
            - static_imputation_model: mean, median, mice, missforest, knn, gain
            - temporal_imputation_model: mean, median, linear, quadratic, cubic, spline, mrnn, tgain

        - feature selection parameters:
            - feature_selection_model: greedy-addtion, greedy-deletion, recursive-addition, recursive-deletion, None
            - feature_number: selected feature number

        - treatment effects model parameters:
            - model_name: CRN, RMSN, GANITE
            Each model has different types of hyperparameters that need to be set.

                - Parameters needed for the Counterfactual Recurrent Network (CRN):
                    - hyperparameters for encoder:
                            - rnn_hidden_units: hidden dimensions in the LSTM unit
                            - rnn_keep_prob: keep probability used for variational dropout in the LSTM unit
                            - br_size: size of the balancing representation
                            - fc_hidden_units: hidden dimensions of the fully connected layers used for treatment classifier and predictor
                            - batch_size: number of samples in mini-batch
                            - num_epochs: number of epochs
                            - learning_rate: learning rate
                            - max_alpha: alpha controls the trade-off between building treatment invariant representations (domain
                                discrimination) and being able to predict outcomes (outcome prediction); during training, CRN uses an
                                exponentially increasing schedule for alpha from 0 to max_alpha.
                    - hyperparameters for decoder:
                            - the decoder requires the same hyperparameters as the encoder with the exception of the rnn_hidden_units
                                which is set to be equal to the br_size of the encoder

                - Parameters for Recurrent Marginal Structural Networks (RMSN):
                        - hyperparameters for encoder:
                                - dropout_rate: dropout probability used for variational
                                - rnn_hidden_units: hidden dimensions in the LSTM unit
                                - batch_size: number of samples in mini-batch
                                - num_epochs: number of epochs
                                - learning_rate: learning rate
                                - max_norm: max gradient norm used for gradient clipping during training
                        - hyperparameters for decoder:
                                - the decoder requires the same hyperparameters as the encoder.
                        - model_dir: directory where the model is saved
                        - model_name: name of the saved model

                - Parameters for GANITE:
                    - batch size: number of samples in mini-batch
                    - alpha: parameter trading off between discriminator loss and supervised loss for the generator training
                    - learning_rate: learning rate
                    - hidden_units: hidden dimensions of the fully connected layers used in the networks
                    - stack_dim: number of timesteps to stack

                All models have the following common parameters:
                    - static_mode: how to utilize static features (concatenate or None)
                    - time_mode: how to utilize time information (concatenate or None)
                    - task: 'classification' or 'regression'


        - metric_name: auc, apr, mae, mse (used for factual prediction)
        - patient id: patient for which counterfactual trajectories are computed
        - timestep: timestep in patient trajectory for estimating counterfactuals
    """
    # %% Step 0: Set basic parameters
    metric_sets = [args.metric_name]
    metric_parameters = {"problem": args.problem, "label_name": [args.label_name]}

    # %% Step 1: Upload Dataset
    # File names
    data_directory = "../datasets/data/" + args.data_name + "/" + args.data_name + "_"

    data_loader_training = CSVLoader(
        static_file=data_directory + "static_train_data.csv.gz",
        temporal_file=data_directory + "temporal_train_data_eav.csv.gz",
    )

    data_loader_testing = CSVLoader(
        static_file=data_directory + "static_test_data.csv.gz",
        temporal_file=data_directory + "temporal_test_data_eav.csv.gz",
    )

    dataset_training = data_loader_training.load()
    dataset_testing = data_loader_testing.load()

    print("Finish data loading.")

    # %% Step 2: Preprocess Dataset
    # (0) filter out negative values (Automatically)
    negative_filter = FilterNegative()
    # (1) one-hot encode categorical features
    onehot_encoder = OneHotEncoder(one_hot_encoding_features=[args.one_hot_encoding])
    # (2) Normalize features: 3 options (minmax, standard, none)
    normalizer = Normalizer(args.normalization)

    filter_pipeline = PipelineComposer(negative_filter, onehot_encoder, normalizer)

    dataset_training = filter_pipeline.fit_transform(dataset_training)
    dataset_testing = filter_pipeline.transform(dataset_testing)

    print("Finish preprocessing.")

    # %% Step 3: Define Problem
    problem_maker = ProblemMaker(
        problem=args.problem, label=[args.label_name], max_seq_len=args.max_seq_len, treatment=[args.treatment]
    )

    dataset_training = problem_maker.fit_transform(dataset_training)
    dataset_testing = problem_maker.fit_transform(dataset_testing)

    print("Finish defining problem.")

    # %% Step 4: Impute Dataset
    static_imputation = Imputation(imputation_model_name=args.static_imputation_model, data_type="static")
    temporal_imputation = Imputation(imputation_model_name=args.temporal_imputation_model, data_type="temporal")

    imputation_pipeline = PipelineComposer(static_imputation, temporal_imputation)

    dataset_training = imputation_pipeline.fit_transform(dataset_training)
    dataset_testing = imputation_pipeline.transform(dataset_testing)

    print("Finish imputation.")

    # %% Step 5: Feature selection (4 options)
    static_feature_selection = FeatureSelection(
        feature_selection_model_name=args.static_feature_selection_model,
        feature_type="static",
        feature_number=args.static_feature_selection_number,
        task=args.task,
        metric_name=args.metric_name,
        metric_parameters=metric_parameters,
    )

    temporal_feature_selection = FeatureSelection(
        feature_selection_model_name=args.temporal_feature_selection_model,
        feature_type="temporal",
        feature_number=args.temporal_feature_selection_number,
        task=args.task,
        metric_name=args.metric_name,
        metric_parameters=metric_parameters,
    )

    feature_selection_pipeline = PipelineComposer(static_feature_selection, temporal_feature_selection)

    dataset_training = feature_selection_pipeline.fit_transform(dataset_training)
    dataset_testing = feature_selection_pipeline.transform(dataset_testing)

    print("Finish feature selection.")

    # %% Step 6: Fit treatment effects (3 options)
    # Set the validation data for best model saving
    dataset_training.train_val_test_split(prob_val=0.2, prob_test=0.0)

    # Set the treatment effects model
    model_name = args.model_name

    # Set treatment effects model parameters
    if model_name == "CRN":
        model_parameters = {
            "encoder_rnn_hidden_units": args.crn_encoder_rnn_hidden_units,
            "encoder_br_size": args.crn_encoder_br_size,
            "encoder_fc_hidden_units": args.crn_encoder_fc_hidden_units,
            "encoder_learning_rate": args.crn_encoder_learning_rate,
            "encoder_batch_size": args.crn_encoder_batch_size,
            "encoder_keep_prob": args.crn_encoder_keep_prob,
            "encoder_num_epochs": args.crn_encoder_num_epochs,
            "encoder_max_alpha": args.crn_encoder_max_alpha,
            "decoder_br_size": args.crn_decoder_br_size,
            "decoder_fc_hidden_units": args.crn_decoder_fc_hidden_units,
            "decoder_learning_rate": args.crn_decoder_learning_rate,
            "decoder_batch_size": args.crn_decoder_batch_size,
            "decoder_keep_prob": args.crn_decoder_keep_prob,
            "decoder_num_epochs": args.crn_decoder_num_epochs,
            "decoder_max_alpha": args.crn_decoder_max_alpha,
            "projection_horizon": args.projection_horizon,
            "static_mode": args.static_mode,
            "time_mode": args.time_mode,
        }
        treatment_model = treatment_effects_model(model_name, model_parameters, task="classification")
        treatment_model.fit(dataset_training)

    elif model_name == "RMSN":
        hyperparams_encoder_iptw = {
            "dropout_rate": args.rmsn_encoder_dropout_rate,
            "memory_multiplier": args.rmsn_encoder_memory_multiplier,
            "num_epochs": args.rmsn_encoder_num_epochs,
            "batch_size": args.rmsn_encoder_batch_size,
            "learning_rate": args.rmsn_encoder_learning_rate,
            "max_norm": args.rmsn_encoder_max_norm,
        }

        hyperparams_decoder_iptw = {
            "dropout_rate": args.rmsn_decoder_dropout_rate,
            "memory_multiplier": args.rmsn_decoder_memory_multiplier,
            "num_epochs": args.rmsn_decoder_num_epochs,
            "batch_size": args.rmsn_decoder_batch_size,
            "learning_rate": args.rmsn_decoder_learning_rate,
            "max_norm": args.rmsn_decoder_max_norm,
        }

        model_parameters = {
            "hyperparams_encoder_iptw": hyperparams_encoder_iptw,
            "hyperparams_decoder_iptw": hyperparams_decoder_iptw,
            "model_dir": args.rmsn_model_dir,
            "model_name": args.rmsn_model_name,
            "static_mode": args.static_mode,
            "time_mode": args.time_mode,
        }

        treatment_model = treatment_effects_model(model_name, model_parameters, task="classification")
        treatment_model.fit(dataset_training, projection_horizon=args.projection_horizon)

    elif model_name == "GANITE":
        hyperparams = {
            "batch_size": args.ganite_batch_size,
            "alpha": args.ganite_alpha,
            "hidden_dims": args.ganite_hidden_dims,
            "learning_rate": args.ganite_learning_rate,
        }

        model_parameters = {
            "hyperparams": hyperparams,
            "stack_dim": args.ganite_stack_dim,
            "static_mode": args.static_mode,
            "time_mode": args.time_mode,
        }

        treatment_model = treatment_effects_model(model_name, model_parameters, task="classification")
        treatment_model.fit(dataset_training)

    test_y_hat = treatment_model.predict(dataset_testing)

    print("Finish treatment effects model training and testing.")

    # %% Step 9: Visualize Results

    # Evaluate predictor model
    result = Metrics(metric_sets, metric_parameters).evaluate(dataset_testing.label, test_y_hat)
    print("Finish predictor model evaluation.")

    # Visualize the output
    # (1) Performance on estimating factual outcomes
    print("Overall performance on estimating factual outcomes")
    print_performance(result, metric_sets, metric_parameters)

    # (2) Counterfactual trajectories
    print("Counterfactual trajectories")
    if model_name in ["CRN", "RMSN"]:
        # Predict and visualize counterfactuals for the sequence of treatments indicated by the user
        # through the treatment_options. The lengths of each sequence of treatments needs to be projection_horizon + 1.
        treatment_options = np.array([[[1], [1], [1], [1], [1], [0]], [[0], [0], [0], [0], [1], [1]]])
        history, counterfactual_traj = treatment_model.predict_counterfactual_trajectories(
            dataset=dataset_testing,
            patient_id=args.patient_id,
            timestep=args.timestep,
            treatment_options=treatment_options,
        )

        print_counterfactual_predictions(
            patient_history=history, treatment_options=treatment_options, counterfactual_predictions=counterfactual_traj
        )

    return


# %%
if __name__ == "__main__":
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_name", choices=["mimic", "ward", "cf", "mimic_antibiotics"], default="mimic_antibiotics", type=str
    )
    parser.add_argument("--normalization", choices=["minmax", "standard", None], default="minmax", type=str)
    parser.add_argument("--one_hot_encoding", default="admission_type", type=str)
    parser.add_argument("--problem", choices=["online", "one-shot"], default="online", type=str)
    parser.add_argument("--max_seq_len", help="maximum sequence length", default=20, type=int)
    parser.add_argument("--label_name", default="ventilator", type=str)
    parser.add_argument("--treatment", default="antibiotics", type=str)
    parser.add_argument(
        "--static_imputation_model",
        choices=["mean", "median", "mice", "missforest", "knn", "gain"],
        default="median",
        type=str,
    )
    parser.add_argument(
        "--temporal_imputation_model",
        choices=["mean", "median", "linear", "quadratic", "cubic", "spline", "mrnn", "tgain"],
        default="median",
        type=str,
    )
    parser.add_argument(
        "--static_feature_selection_model",
        choices=["greedy-addition", "greedy-deletion", "recursive-addition", "recursive-deletion", None],
        default=None,
        type=str,
    )
    parser.add_argument("--static_feature_selection_number", default=10, type=int)
    parser.add_argument(
        "--temporal_feature_selection_model",
        choices=["greedy-addition", "greedy-deletion", "recursive-addition", "recursive-deletion", None],
        default=None,
        type=str,
    )
    parser.add_argument("--temporal_feature_selection_number", default=10, type=int)
    parser.add_argument("--model_name", choices=["CRN", "RMSN", "GANITE"], default="CRN", type=str)
    parser.add_argument("--projection_horizon", default=5, type=int)

    # Hyperparams for CRN
    parser.add_argument("--crn_encoder_rnn_hidden_units", default=128, type=int)
    parser.add_argument("--crn_encoder_br_size", default=64, type=int)
    parser.add_argument("--crn_encoder_fc_hidden_units", default=128, type=int)
    parser.add_argument("--crn_encoder_learning_rate", default=0.001, type=int)
    parser.add_argument("--crn_encoder_batch_size", default=256, type=int)
    parser.add_argument("--crn_encoder_keep_prob", default=0.9, type=int)
    parser.add_argument("--crn_encoder_num_epochs", default=100, type=float)
    parser.add_argument("--crn_encoder_max_alpha", default=1.0, type=float)

    parser.add_argument("--crn_decoder_br_size", default=64, type=int)
    parser.add_argument("--crn_decoder_fc_hidden_units", default=128, type=int)
    parser.add_argument("--crn_decoder_learning_rate", default=0.001, type=int)
    parser.add_argument("--crn_decoder_batch_size", default=512, type=int)
    parser.add_argument("--crn_decoder_keep_prob", default=0.9, type=int)
    parser.add_argument("--crn_decoder_num_epochs", default=100, type=float)
    parser.add_argument("--crn_decoder_max_alpha", default=1.0, type=float)

    # Hyperparams for RMSN
    parser.add_argument("--rmsn_encoder_dropout_rate", default=0.1, type=int)
    parser.add_argument("--rmsn_encoder_memory_multiplier", default=4.0, type=int)
    parser.add_argument("--rmsn_encoder_num_epochs", default=100, type=int)
    parser.add_argument("--rmsn_encoder_batch_size", default=64, type=int)
    parser.add_argument("--rmsn_encoder_learning_rate", default=0.01, type=int)
    parser.add_argument("--rmsn_encoder_max_norm", default=0.5, type=int)
    parser.add_argument("--rmsn_decoder_dropout_rate", default=0.1, type=float)
    parser.add_argument("--rmsn_decoder_memory_multiplier", default=2.0, type=float)
    parser.add_argument("--rmsn_decoder_num_epochs", default=100, type=int)
    parser.add_argument("--rmsn_decoder_batch_size", default=512, type=int)
    parser.add_argument("--rmsn_decoder_learning_rate", default=0.001, type=int)
    parser.add_argument("--rmsn_decoder_max_norm", default=4.0, type=int)
    parser.add_argument("--rmsn_model_dir", default="tmp/", type=str)
    parser.add_argument("--rmsn_model_name", default="rmsn", type=str)

    # Hyperparams for GANITE
    parser.add_argument("--ganite_batch_size", default=128, type=float)
    parser.add_argument("--ganite_alpha", default=1.0, type=int)
    parser.add_argument("--ganite_hidden_dims", default=128, type=int)
    parser.add_argument("--ganite_learning_rate", default=0.001, type=int)
    parser.add_argument("--ganite_stack_dim", default=4, type=int)

    parser.add_argument("--patient_id", default=2, type=int)

    parser.add_argument("--timestep", default=5, type=int)

    parser.add_argument("--static_mode", choices=["concatenate", None], default="concatenate", type=str)
    parser.add_argument("--time_mode", choices=["concatenate", None], default="concatenate", type=str)
    parser.add_argument("--task", choices=["classification", "regression"], default="classification", type=str)
    parser.add_argument("--metric_name", choices=["auc", "apr", "mse", "mae"], default="auc", type=str)

    args = parser.parse_args()

    # Call main function
    main(args)
