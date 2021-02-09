"""Make a model ensemble based on AutoML results.

References: Y. Zhang, D. Jarrett, M. van der Schaar, 
                     "Stepwise Model Selection for Sequence Prediction via Deep Kernel Learning," 
                     International Conference on Artificial Intelligence and Statistics (AISTATS), 2020.
"""

# Necessary packages
import numpy as np
from base import BaseEstimator


class AutoEnsemble(BaseEstimator):
    """Making a model ensemble based on AutoML results.

    This model can only be created using the results of AutoML. 
    It therefore does not have a `fit` method.
    
    Attributes:
        - models: predictive models
        - bo_score: bayesian optimization scores for each model for each time
    """

    def __init__(self, models, bo_score):

        model_time = np.argmin(bo_score, axis=0)
        all_models = np.unique(model_time)
        time_dict = dict(zip(all_models, range(len(all_models))))
        self.model_ind = [time_dict[i] for i in model_time]
        self.models = [models[x] for x in all_models]
        self.task = models[0].task
        self.static_mode = models[0].static_mode
        self.time_mode = models[0].time_mode

    def fit(self, dataset):
        raise ValueError("Cannot fit ensemble directly. Using automl package instead.")

    def predict(self, dataset, **kwargs):
        """Making prediction with the ensemble.

        Args:
            - dataset: The input :class:`~datasets.PandasDataset` object with problem defined consistently with the model.
            - **kwargs: Additional arguments to be passed to the predict method of the base models in the ensemble.

        Returns:
            - pred: Predictions.
        """
        predictions = [x.predict(dataset, **kwargs) for x in self.models]

        if dataset.problem != "one-shot":
            res_list = list()
            for i in range(len(self.model_ind)):
                res = predictions[self.model_ind[i]][:, i : i + 1, :]
                res_list.append(res)
            pred = np.concatenate(res_list, axis=1)
        else:
            pred = predictions[0]
        return pred
