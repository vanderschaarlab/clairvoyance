"""Performance metrics.

(1) Metrics: General metric class
    (a) Online problem
    - Performance for each timestamp for each sequence
    (b) One-shot problem
    - Performance for each sequence
    
(2) BOMetrics: Metric class for Bayesian Optimization
"""

# Necessary packages
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error


class Metrics:
    """General metric class.
    
    Attributes:
        - metric_sets: sets of performance metrics (e.g. ['auc','apr'])
        - metric_parameters: problem definition and label names
    """

    def __init__(self, metric_sets, metric_parameters):
        self.problem = metric_parameters["problem"]
        self.label_sets = metric_parameters["label_name"]
        self.metric_sets = metric_sets

    def metric_each(self, y, y_hat, metric_name):
        """Compute metrics for each metric name (AUC, APR, MSE, MAE).
        
        Args:
            y: labels
            y_hat: predictions
            metric_name: type of metrics
            
        Returns:
            result: computed performance metric.
        """
        # Metrics should be among 'auc', 'apr', 'mse', 'mae'
        assert metric_name in ["auc", "apr", "mse", "mae"]
        # Initialize the output
        result = 0
        # Exclude padding part
        idx = np.where(y >= 0)[0]
        y = y[idx]
        y_hat = y_hat[idx]
        # Compute performances
        if metric_name == "auc":
            if len(np.unique(y)) == 2:
                y = y.astype(bool)
                result = roc_auc_score(y, y_hat)
        elif metric_name == "apr":
            if len(np.unique(y)) == 2:
                y = y.astype(bool)
                result = average_precision_score(y, y_hat)
        elif metric_name == "mse":
            result = mean_squared_error(y, y_hat)
        elif metric_name == "mae":
            result = mean_absolute_error(y, y_hat)

        return result

    def evaluate(self, y, y_hat):
        """Returns the prediction performance.
        
        Args:
            - y: labels
            - y_hat: predictions
            
        Returns:
            - result: computed performances
        """
        # Initialize the output
        result = dict()
        # For each label (handling multi-label settings)
        for l_idx in range(len(self.label_sets)):
            # For each metric
            for m_idx in range(len(self.metric_sets)):
                # Online
                if self.problem == "online":
                    result_temp = np.zeros([y.shape[1]])
                    # For each time stamp
                    for t_idx in range(y.shape[1]):
                        result_temp[t_idx] = self.metric_each(
                            y[:, t_idx, l_idx], y_hat[:, t_idx, l_idx], self.metric_sets[m_idx]
                        )
                # One-shot
                elif self.problem == "one-shot":
                    result_temp = self.metric_each(y[:, l_idx], y_hat[:, l_idx], self.metric_sets[m_idx])
                # Save results for each metric and label
                result[self.label_sets[l_idx] + " + " + self.metric_sets[m_idx]] = result_temp

        return result


class BOMetric:
    """Metric class used for Bayesian Optimization.
    
    Attributes:
        - metric: metric_sets: performance metric sets (e.g. ['auc','apr'])
        - fold: The fold in the data set to evaluate model.
        - split: The split in the data set to evaluate model.
    """

    def __init__(self, metric, fold=0, split="test"):
        self.fold = fold
        self.split = split
        acc_list = ["auc", "apr"]
        self.metric_sets = [metric]
        if metric in acc_list:
            self.multiplier = -1.0
        else:
            self.multiplier = 1.0

    def eval(self, dataset, y_hat):
        """Evaluate model prediction.
        
        The accuracy metrics (e.g. auc) will be multiplied by `-1` and presented as a "loss".
        
        Args:
            - dataset: A :class:`~datasets.PandasDataset` with problem defined.
            - y_hat: The predicted values
        
        Returns:
            - out: The evaluation metric at each time step as a numpy 2D array with shape `[1, n_time_step]`.
        """
        # Define parameters
        problem = dataset.problem
        label_name = dataset.label_name
        metric_parameters = {"problem": problem, "label_name": label_name}
        # Get labels
        _, _, fold_label, _, _ = dataset.get_fold(self.fold, self.split)
        # Compute performance
        metric_class = Metrics(self.metric_sets, metric_parameters)
        result = metric_class.evaluate(fold_label, y_hat)
        if metric_parameters["problem"] == "online":
            res_arr = np.stack(list(result.values()))
            res_arr[res_arr == 0.0] = np.nan
            out = self.multiplier * np.nanmean(res_arr, axis=0)[None, :]
            out[np.isnan(out)] = 0.0
        else:
            out = np.nanmean(np.array(list(result.values())))
            out = np.zeros((1, dataset.temporal_feature.shape[1]), dtype=np.float) + out
            out = self.multiplier * out
        return out
