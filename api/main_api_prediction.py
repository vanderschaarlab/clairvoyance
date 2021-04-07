"""Main function for time-series prediction.

Pipeline
Step 1: Load dataset
    - data_name: mimic, ward, cf
    
Step 2: Preprocess dataset
    (0) NegativeFilter: Replace negative values to NaN
    (1) OneHotEncoder: One hot encoding certain features
    (2) Normalization (3 options): MinMax, Standard, None
    
Step 3: Define problem
    - problem: one-shot or online
    - label_name: the column name for the label(s)
    - max_seq_len: maximum sequence length after padding
    - treatment: the column name for treatments
    
Step 4: Impute dataset
    (0) Static imputation (6 options): mean, median, mice, missforest, knn, gain
    (1) Temporal imputation (8 options): mean, median, linear, quadratic, cubic, spline, mrnn, tgain
    
Step 5: Feature selection
    - feature selection method (5 options): greedy-addition, greedy-deletion, recursive-addition, recursive-deletion, None
    
Step 6: Fit and Predict
    - predictive models (6 options): lstm, gru, rnn, attention, tcn, transformer
    
Step 7: Estimate uncertainty (1 option)

Step 8: Interpret predictions (1 option)

Step 9: Visualize Results
    - metric_name (4 options): auc, apr, mse, mae
    (1) Visualize the performance
    (2) Visualize the predictions
    (3) Visualize the uncertainty
    (4) Visualize the interpratations
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
from prediction import prediction
from uncertainty import uncertainty
from interpretation import interpretation
from evaluation import Metrics
from evaluation import print_performance, print_prediction, print_uncertainty, print_interpretation
from utils import PipelineComposer


def main(args):
    """Main function for time-series prediction.
    
    Args:
        - data loading parameters:
            - data_names: mimic, ward, cf        
            
        - preprocess parameters: 
            - normalization: minmax, standard, None
            - one_hot_encoding: input features that need to be one-hot encoded
            - problem: 'one-shot' or 'online'
                - 'one-shot': one time prediction at the end of the time-series 
                - 'online': prediction at every time stamps of the time-series
            - max_seq_len: maximum sequence length after padding
            - label_name: the column name for the label(s)
            - treatment: the column name for treatments
            
        - imputation parameters: 
            - static_imputation_model: mean, median, mice, missforest, knn, gain
            - temporal_imputation_model: mean, median, linear, quadratic, cubic, spline, mrnn, tgain
                        
        - feature selection parameters:
            - feature_selection_model: greedy-addition, greedy-deletion, recursive-addition, recursive-deletion, None
            - feature_number: selected feature number
            
        - predictor_parameters:
            - model_name: rnn, gru, lstm, attention, tcn, transformer
            - model_parameters: network parameters such as number of layers
                - h_dim: hidden dimensions
                - n_layer: layer number
                - n_head: head number (only for transformer model)
                - batch_size: number of samples in mini-batch
                - epochs: number of epochs
                - learning_rate: learning rate
            - static_mode: how to utilize static features (concatenate or None)
            - time_mode: how to utilize time information (concatenate or None)
            - task: classification or regression
            
        - uncertainty_model_name: uncertainty estimation model name (ensemble)
        - interpretation_model_name: interpretation model name (tinvase)
        - metric_name: auc, apr, mae, mse
    """
    #%% Step 0: Set basic parameters
    metric_sets = [args.metric_name]
    metric_parameters = {"problem": args.problem, "label_name": [args.label_name]}

    #%% Step 1: Upload Dataset
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

    #%% Step 2: Preprocess Dataset
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

    #%% Step 3: Define Problem
    problem_maker = ProblemMaker(
        problem=args.problem, label=[args.label_name], max_seq_len=args.max_seq_len, treatment=args.treatment
    )

    dataset_training = problem_maker.fit_transform(dataset_training)
    dataset_testing = problem_maker.fit_transform(dataset_testing)

    print("Finish defining problem.")

    #%% Step 4: Impute Dataset
    static_imputation = Imputation(imputation_model_name=args.static_imputation_model, data_type="static")
    temporal_imputation = Imputation(imputation_model_name=args.temporal_imputation_model, data_type="temporal")

    imputation_pipeline = PipelineComposer(static_imputation, temporal_imputation)

    dataset_training = imputation_pipeline.fit_transform(dataset_training)
    dataset_testing = imputation_pipeline.transform(dataset_testing)

    print("Finish imputation.")

    #%% Step 5: Feature selection (4 options)
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

    #%% Step 6: Fit and Predict (6 options)
    # Set predictor model parameters
    model_parameters = {
        "h_dim": args.h_dim,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "batch_size": args.batch_size,
        "epoch": args.epochs,
        "model_type": args.model_name,
        "learning_rate": args.learning_rate,
        "static_mode": args.static_mode,
        "time_mode": args.time_mode,
        "verbose": True,
    }

    # Set the validation data for best model saving
    dataset_training.train_val_test_split(prob_val=0.2, prob_test=0.0)

    pred_class = prediction(args.model_name, model_parameters, args.task)
    pred_class.fit(dataset_training)
    test_y_hat = pred_class.predict(dataset_testing)

    print("Finish predictor model training and testing.")

    #%% Step 7: Estimate Uncertainty (1 option)
    uncertainty_model = uncertainty(args.uncertainty_model_name, model_parameters, pred_class, args.task)
    uncertainty_model.fit(dataset_training)
    test_ci_hat = uncertainty_model.predict(dataset_testing)
    print("Finish uncertainty estimation")

    #%% Step 8: Interpret Predictions (1 option)
    interpretor = interpretation(args.interpretation_model_name, model_parameters, pred_class, args.task)
    interpretor.fit(dataset_training)
    test_s_hat = interpretor.predict(dataset_testing)
    print("Finish model interpretation")

    #%% Step 9: Visualize Results
    idx = np.random.permutation(len(test_y_hat))[:2]

    # Evaluate predictor model
    result = Metrics(metric_sets, metric_parameters).evaluate(dataset_testing.label, test_y_hat)
    print("Finish predictor model evaluation.")

    # Visualize the output
    # (1) Performance
    print("Overall performance")
    print_performance(result, metric_sets, metric_parameters)
    # (2) Predictions
    print("Each prediction")
    print_prediction(test_y_hat[idx], metric_parameters)
    # (3) Uncertainty
    print("Uncertainty estimations")
    print_uncertainty(test_y_hat[idx], test_ci_hat[idx], metric_parameters)
    # (4) Model interpretation
    print("Model interpretation")
    print_interpretation(test_s_hat[idx], dataset_training.feature_name, metric_parameters, model_parameters)

    return


#%%
if __name__ == "__main__":

    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", choices=["mimic", "ward", "cf"], default="cf", type=str)
    parser.add_argument("--normalization", choices=["minmax", "standard", None], default="minmax", type=str)
    parser.add_argument("--one_hot_encoding", default="admission_type", type=str)
    parser.add_argument("--problem", choices=["online", "one-shot"], default="one-shot", type=str)
    parser.add_argument("--max_seq_len", help="maximum sequence length", default=24, type=int)
    parser.add_argument("--label_name", default="death", type=str)
    parser.add_argument("--treatment", default=None, type=str)
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
    parser.add_argument(
        "--model_name", choices=["rnn", "gru", "lstm", "attention", "tcn", "transformer"], default="attention", type=str
    )
    parser.add_argument("--h_dim", default=100, type=int)
    parser.add_argument("--n_layer", default=2, type=int)
    parser.add_argument("--n_head", default=2, type=int)
    parser.add_argument("--batch_size", default=400, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--static_mode", choices=["concatenate", None], default="concatenate", type=str)
    parser.add_argument("--time_mode", choices=["concatenate", None], default="concatenate", type=str)
    parser.add_argument("--task", choices=["classification", "regression"], default="classification", type=str)
    parser.add_argument("--uncertainty_model_name", choices=["ensemble"], default="ensemble", type=str)
    parser.add_argument("--interpretation_model_name", choices=["tinvase"], default="tinvase", type=str)
    parser.add_argument("--metric_name", choices=["auc", "apr", "mse", "mae"], default="auc", type=str)

    args = parser.parse_args()

    # Call main function
    main(args)
