"""Main function for AutoML in time-series predictions.

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
  - feature selection method (5 options): greedy-addtion, greedy-deletion, recursive-addition, recursive-deletion, None
  
Step 6: Fit and Predict AutoML
  - possible predictive models (6 options): lstm, gru, rnn, attention, tcn, transformer

Step 7: Visualize Results
  - metric_name (4 options): auc, apr, mse, mae
  (1) Visualize the performance
  (2) Visualize the predictions
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
from prediction import GeneralRNN, TemporalCNN, TransformerPredictor, Attention
from prediction import AutoEnsemble, StackingEnsemble
import automl
from evaluation import Metrics, BOMetric
from evaluation import print_performance, print_prediction
from utils import PipelineComposer


def main(args):
    """Main function for AutoML in time-series predictions.
  
  Args:
    - data loading parameters:
      - data_names: mimic, ward, cf    
      
    - preprocess parameters: 
      - normalization: minmax, standard, None
      - one_hot_encoding: input features that need to be one-hot encoded
      - problem: 'one-shot' or 'online'
        - 'one-shot': one time prediction at the end of the time-series 
        - 'online': preditcion at every time stamps of the time-series
      - max_seq_len: maximum sequence length after padding
      - label_name: the column name for the label(s)
      - treatment: the column name for treatments
      
    - imputation parameters: 
      - static_imputation_model: mean, median, mice, missforest, knn, gain
      - temporal_imputation_model: mean, median, linear, quadratic, cubic, spline, mrnn, tgain
            
    - feature selection parameters:
      - feature_selection_model: greedy-addtion, greedy-deletion, recursive-addition, recursive-deletion, None
      - feature_number: selected featuer number
      
    - predictor_parameters:
      - epochs: number of epochs
      - bo_itr: bayesian optimization iterations
      - static_mode: how to utilize static features (concatenate or None)
      - time_mode: how to utilize time information (concatenate or None)
      - task: classification or regression
      
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

    #%% Step 6: Bayesian Optimization
    ## Model define
    # RNN model
    rnn_parameters = {
        "model_type": "lstm",
        "epoch": args.epochs,
        "static_mode": args.static_mode,
        "time_mode": args.time_mode,
        "verbose": False,
    }

    general_rnn = GeneralRNN(task=args.task)
    general_rnn.set_params(**rnn_parameters)

    # CNN model
    cnn_parameters = {
        "epoch": args.epochs,
        "static_mode": args.static_mode,
        "time_mode": args.time_mode,
        "verbose": False,
    }
    temp_cnn = TemporalCNN(task=args.task)
    temp_cnn.set_params(**cnn_parameters)

    # Transformer
    transformer = TransformerPredictor(
        task=args.task, epoch=args.epochs, static_mode=args.static_mode, time_mode=args.time_mode
    )

    # Attention model
    attn_parameters = {
        "model_type": "lstm",
        "epoch": args.epochs,
        "static_mode": args.static_mode,
        "time_mode": args.time_mode,
        "verbose": False,
    }
    attn = Attention(task=args.task)
    attn.set_params(**attn_parameters)

    # model_class_list = [general_rnn, attn, temp_cnn, transformer]
    model_class_list = [general_rnn, attn]

    # train_validate split
    dataset_training.train_val_test_split(prob_val=0.2, prob_test=0.1)

    # Bayesian Optimization Start
    metric = BOMetric(metric="auc", fold=0, split="test")

    ens_model_list = []

    # Run BO for each model class
    for m in model_class_list:
        BO_model = automl.model.AutoTS(dataset_training, m, metric, model_path="tmp/")
        models, bo_score = BO_model.training_loop(num_iter=args.bo_itr)
        auto_ens_model = AutoEnsemble(models, bo_score)
        ens_model_list.append(auto_ens_model)

    # Load all ensemble models
    for ens in ens_model_list:
        for m in ens.models:
            m.load_model(BO_model.model_path + "/" + m.model_id + ".h5")

    # Stacking algorithm
    stacking_ens_model = StackingEnsemble(ens_model_list)
    stacking_ens_model.fit(dataset_training, fold=0, train_split="val")

    # Prediction
    assert not dataset_testing.is_validation_defined
    test_y_hat = stacking_ens_model.predict(dataset_testing, test_split="test")
    test_y = dataset_testing.label

    print("Finish AutoML model training and testing.")

    #%% Step 7: Visualize Results
    idx = np.random.permutation(len(test_y_hat))[:2]

    # Evaluate predictor model
    result = Metrics(metric_sets, metric_parameters).evaluate(test_y, test_y_hat)
    print("Finish predictor model evaluation.")

    # Visualize the output
    # (1) Performance
    print("Overall performance")
    print_performance(result, metric_sets, metric_parameters)
    # (2) Predictions
    print("Each prediction")
    print_prediction(test_y_hat[idx], metric_parameters)

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
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--bo_itr", default=20, type=int)
    parser.add_argument("--static_mode", choices=["concatenate", None], default="concatenate", type=str)
    parser.add_argument("--time_mode", choices=["concatenate", None], default="concatenate", type=str)
    parser.add_argument("--task", choices=["classification", "regression"], default="classification", type=str)
    parser.add_argument("--metric_name", choices=["auc", "apr", "mse", "mae"], default="auc", type=str)

    args = parser.parse_args()

    # Call main function
    main(args)
