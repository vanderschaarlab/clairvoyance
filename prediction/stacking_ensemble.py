"""Ensemble models using Stacking. 
"""

# Necessary packages
import torch
import torch.nn as nn
from base import BaseEstimator, PredictorMixin


class StackingEnsemble(BaseEstimator, PredictorMixin):
    """Construct the ensemble models using stacking.
    
    Attributes:
        - model_list: list of predictive models for ensembling
    """

    def __init__(self, model_list):

        super().__init__()

        self.model_list = model_list
        # Set parameters
        self.task = model_list[0].task
        self.static_mode = model_list[0].static_mode
        self.time_mode = model_list[0].time_mode

        self.weight_param = nn.Parameter(torch.ones(len(model_list)), requires_grad=True)
        self.optimizer = torch.optim.Adam([self.weight_param], lr=0.01)

    def loss_fn(self, y_hat, y):
        """Loss functions (Cross entropy for classification, RMSE for regression).
        
        Args:
            - y_hat: predictions
            - y: labels
            
        Returns:
            - loss: computed loss
        """
        if self.task == "classification":
            loss = torch.sum(torch.log(y_hat + 1e-9) * y + torch.log(1.0 - y_hat + 1e-9) * (1.0 - y)) * -1.0
        else:
            loss = torch.sqrt(torch.mean((y_hat - y) ** 2))
        return loss

    def _make_tensor(self, x):
        """Construct tensor.
        """
        return torch.tensor(x, dtype=torch.float)

    def fit(self, dataset, fold=0, train_split="val"):
        """Train the ensemble model.
        
        Args:
            - dataset: training data
            - fold: data fold number
            - train_split: 'train' or 'val'
        """
        preds = [x.predict(dataset, fold=fold, test_split=train_split) for x in self.model_list]
        preds = [self._make_tensor(x) for x in preds]

        # Set labels
        _, y = self._data_preprocess(dataset, fold, train_split)
        y = self._make_tensor(y)

        # B x D x M
        pred_mat = torch.stack(preds, dim=-1)

        for i in range(1000):
            weights = nn.functional.softmax(self.weight_param)
            y_hat = torch.matmul(pred_mat, weights)

            if self.task == "classification":
                loss = torch.sum(torch.log(y_hat + 1e-9) * y + torch.log(1.0 - y_hat + 1e-9) * (1.0 - y)) * -1.0
            else:
                loss = torch.sqrt(torch.mean((y_hat - y) ** 2))

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def predict(self, dataset, fold=0, test_split="test"):
        """Return the predictions using trained model.
        
        Args:
            - dataset: training data
            - fold: data fold number
            - test_split: test splitting settings
            
        Returns:
            - y_hat: predictions
        """
        preds = [x.predict(dataset, fold=fold, test_split=test_split) for x in self.model_list]
        preds = [self._make_tensor(x) for x in preds]
        pred_mat = torch.stack(preds, dim=-1)
        # Stacking
        with torch.no_grad():
            weights = nn.functional.softmax(self.weight_param)
            y_hat = torch.matmul(pred_mat, weights).cpu().numpy()
        return y_hat

    def save_model(self, model_path):
        # TODO: recursively save all models and then save weights
        pass

    def load_model(self, model_path):
        # TODO: recursively load all models and create class
        pass
