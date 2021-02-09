"""Feature selection in time-series setting.

(1) model_predict_and_evaluate: train model with subset of the features and evaluate the performance

(2) greedy_feature_selection: select subset of the features in greedy way
    - addition: Top k features with best performances only with that feature
    - deletion: Top k features with worst performance without that feature
    
(3) recursive_feature_selection: select subset of the features in recursive way
    - addition: Recursively select top k features with greedy feature addition
    - deletion: Recursively select top k features with greedy feature deletion
"""

# Necessary packages
import numpy as np
from tqdm import tqdm
from utils import list_diff
from evaluation import Metrics
from prediction import prediction


class FeatureSelection:
    """Feature selection class.
    
    Attributes:        
        - feature_selection_model_name: 'greedy-addition', 'greedy-deletion', 'recursive-addition', 'recursive-deletion', None
        - feature_type: 'static' or 'temporal'
        - feature_number: the number of selected features
        - task: classification or regression
        - metric_name: 'auc', 'apr', 'mae', 'mse'
        - metric_parameters: problem setting and labels
        - model_parameters: parameters for training model        
    """

    def __init__(
        self,
        feature_selection_model_name,
        feature_type,
        feature_number,
        task,
        metric_name,
        metric_parameters,
        model_parameters=None,
    ):

        self.feature_selection_model_name = feature_selection_model_name
        self.feature_type = feature_type
        self.feature_number = feature_number
        self.task = task
        self.metric_name = metric_name
        self.metric_parameters = metric_parameters
        self.model_parameters = model_parameters
        # If not specified, use GRU.
        if self.model_parameters == None:
            self.model_parameters = {
                "model_type": "gru",
                "h_dim": 10,
                "n_head": 2,
                "n_layer": 2,
                "batch_size": 100,
                "epoch": 20,
                "learning_rate": 0.001,
                "static_mode": "concatenate",
                "time_mode": "concatenate",
                "verbose": True,
            }

        # Output initialization
        self.selected_features = None

    def model_predict_and_evaluate(self, dataset):
        """Train model with subset of the features and evaluate the performance.
        
        Args:
            - dataset: dataset with subset of temporal features
            
        Returns:
            - performance: performance with subset of temporal features
        """
        # Build model
        pred_class = prediction(self.model_parameters["model_type"], self.model_parameters, self.task)
        # Train the model
        pred_class.fit(dataset)
        # Test the model
        test_y_hat = pred_class.predict(dataset)
        # Extract the labels
        _, _, test_y, _, _ = dataset.get_fold(fold=0, split="test")
        # Evaluate the performance
        temp_performance = Metrics([self.metric_name], self.metric_parameters).evaluate(test_y, test_y_hat)
        performance = np.mean(list(temp_performance.values())[0])

        return performance

    def greedy_feature_selection(self, dataset, feature_selection_model):
        """Select subset of the features in greedy way.
        
        Args:
            - dataset: dataset with subset of temporal features
            - feature_selection_model: 'addition' or 'deletion'
            
        Returns:
            - selected_feature_index: selected feature index    
        """
        assert feature_selection_model in ["addition", "deletion"]
        if self.model_parameters["static_mode"] != "concatenate":
            assert self.feature_type == "temporal"

        # Save original data
        if self.feature_type == "temporal":
            ori_temporal_feature = dataset.temporal_feature.copy()
            # Parameters
            no, seq_len, dim = ori_temporal_feature.shape
        elif self.feature_type == "static":
            ori_static_feature = dataset.static_feature.copy()
            # Parameters
            no, dim = ori_static_feature.shape

        ## Initialization
        # Entire feature set
        feature_set = [i for i in range(dim)]
        # Save results in dictionary
        result = dict()

        # Greedy way of evaluating the importance of each feature
        for f in tqdm(feature_set):
            # For addition option, just select certain feature
            if feature_selection_model == "addition":
                temp_feature = [f]
            # For deletion option, remove certain feature from the entire feature set
            elif feature_selection_model == "deletion":
                temp_feature = list_diff(feature_set, [f])

            # Set the temporal data only with the subset of features
            if self.feature_type == "temporal":
                dataset.temporal_feature = ori_temporal_feature[:, :, temp_feature].copy()
            elif self.feature_type == "static":
                dataset.static_feature = ori_static_feature[:, temp_feature].copy()

            # Model train and test
            result[f] = self.model_predict_and_evaluate(dataset)
            # For deletion, worse performance represents better.
            if feature_selection_model == "deletion":
                result[f] = -result[f]

        # Select top feature_number features by result
        if self.metric_name in ["auc", "apr"]:
            selected_feature_index = sorted(result, key=result.get, reverse=True)[: self.feature_number]
        elif self.metric_name in ["mse", "mae"]:
            selected_feature_index = sorted(result, key=result.get, reverse=False)[: self.feature_number]

        # Recover the original data
        if self.feature_type == "temporal":
            dataset.temporal_feature = ori_temporal_feature
        elif self.feature_type == "static":
            dataset.static_feature = ori_static_feature

        return selected_feature_index

    def recursive_feature_selection(self, dataset, feature_selection_model):
        """Select subset of the features in recursive way.
        
        Args:
            - dataset: dataset with subset of temporal features
            - feature_selection_model: 'addition' or 'deletion'
            
        Returns:
            - curr_set: selected feature index    
        """
        assert feature_selection_model in ["addition", "deletion"]
        if self.model_parameters["static_mode"] != "concatenate":
            assert self.feature_type == "temporal"

        # Save original data
        if self.feature_type == "temporal":
            ori_temporal_feature = dataset.temporal_feature.copy()
            # Parameters
            no, seq_len, dim = ori_temporal_feature.shape
        elif self.feature_type == "static":
            ori_static_feature = dataset.static_feature.copy()
            # Parameters
            no, dim = ori_static_feature.shape

        ## Initialization
        # Entire feature set
        feature_set = [i for i in range(dim)]

        # current feature set.
        # Deletion starts from the entire feature set
        if feature_selection_model == "deletion":
            curr_set = [i for i in range(dim)]
            iterations = dim - self.feature_number
        # Addition starts from empty set
        elif feature_selection_model == "addition":
            curr_set = list()
            iterations = self.feature_number

        # Iterate until the number of selected features = feature_number
        for i in tqdm(range(iterations)):
            # Save results in dictionary
            result = dict()
            # For each feature
            for f in tqdm(feature_set):
                # For addition option, just select certain feature
                if feature_selection_model == "addition":
                    temp_feature = curr_set + [f]
                # For deletion option, remove certain feature from the entire feature set
                elif feature_selection_model == "deletion":
                    temp_feature = list_diff(curr_set, [f])

                # Set the temporal data only with the subset of features
                if self.feature_type == "temporal":
                    dataset.temporal_feature = ori_temporal_feature[:, :, temp_feature].copy()
                elif self.feature_type == "static":
                    dataset.static_feature = ori_static_feature[:, temp_feature].copy()

                # Model training and testing
                result[f] = self.model_predict_and_evaluate(dataset)

            # Find the feature with best performance
            if self.metric_name in ["auc", "apr"]:
                temp_feature_index = max(result, key=result.get)
            elif self.metric_name in ["mse", "mae"]:
                temp_feature_index = min(result, key=result.get)

            # Recursively set the current feature set
            if feature_selection_model == "deletion":
                curr_set = list_diff(curr_set, [temp_feature_index])
            if feature_selection_model == "addition":
                curr_set = curr_set + [temp_feature_index]

            # Remove the selected feature from the entire feature set
            feature_set = list_diff(feature_set, [temp_feature_index])

        # Recover the original data
        if self.feature_type == "temporal":
            dataset.temporal_feature = ori_temporal_feature
        elif self.feature_type == "static":
            dataset.static_feature = ori_static_feature

        return curr_set

    def fit(self, dataset):
        """Fit feature selection module.
        
        Args:
            - dataset: original input data
        """
        assert self.feature_selection_model_name in [
            "greedy-addition",
            "greedy-deletion",
            "recursive-addition",
            "recursive-deletion",
            None,
        ]
        if self.feature_selection_model_name == "greedy-addition":
            self.selected_features = self.greedy_feature_selection(dataset, "addition")
        elif self.feature_selection_model_name == "greedy-deletion":
            self.selected_features = self.greedy_feature_selection(dataset, "deletion")
        elif self.feature_selection_model_name == "recursive-addition":
            self.selected_features = self.recursive_feature_selection(dataset, "addition")
        elif self.feature_selection_model_name == "recursive-deletion":
            self.selected_features = self.recursive_feature_selection(dataset, "deletion")

        return

    def transform(self, dataset):
        """Remain only selected features.
        
        Args:
            - dataset: original input data
            
        Returns:
            - dataset: dataset only with selected features
        """
        if self.selected_features is not None:
            if self.feature_type == "static":
                dataset.static_feature = dataset.static_feature[:, self.selected_features]
                temp_feature_name = [dataset.feature_name["static"][i] for i in self.selected_features]
                dataset.feature_name["static"] = temp_feature_name
            elif self.feature_type == "temporal":
                dataset.temporal_feature = dataset.temporal_feature[:, :, self.selected_features]
                temp_feature_name = [dataset.feature_name["temporal"][i] for i in self.selected_features]
                dataset.feature_name["temporal"] = temp_feature_name

        return dataset

    def fit_transform(self, dataset):
        """Fit and transform. Return dataset only with selected features
        
        Args:
            - dataset: original dataset
        """
        self.fit(dataset)
        return self.transform(dataset)
