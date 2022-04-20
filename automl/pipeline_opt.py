from GPyOpt.methods import BayesianOptimization
import numpy as np
from utils import PipelineComposer
from evaluation import Metrics, BOMetric
import imputation
from feature_selection import FeatureSelection
from prediction import prediction

# from evaluation import print_performance


class PipelineOpt:
    def __init__(
        self,
        static_imputation_model_list,
        temporal_imputation_model_list,
        static_feature_selection_model_list,
        temporal_feature_selection_model_list,
        prediction_model_list,
        dataset_training,
        dataset_testing,
        task,
        metric_name,
        metric_parameters,
    ):
        self.dataset_testing = dataset_testing
        self.dataset_training = dataset_training
        self.static_imputation_model_list = static_imputation_model_list
        self.temporal_imputation_model_list = temporal_imputation_model_list
        self.static_feature_selection_model_list = static_feature_selection_model_list
        self.temporal_feature_selection_model_list = temporal_feature_selection_model_list
        self.prediction_model_list = prediction_model_list

        # imputation

        static_imputation_list = [
            imputation.Imputation(imputation_model_name=x, data_type="static") for x in static_imputation_model_list
        ]
        temporal_imputation_list = [
            imputation.Imputation(imputation_model_name=x, data_type="temporal") for x in temporal_imputation_model_list
        ]

        # feature selection

        static_feature_selection_list = []
        for x in static_feature_selection_model_list:
            # Select relevant features
            static_feature_selection = FeatureSelection(
                feature_selection_model_name=x[0],
                feature_type="static",
                feature_number=x[1],
                task=task,
                metric_name=metric_name,
                metric_parameters=metric_parameters,
            )
            static_feature_selection_list.append(static_feature_selection)

        temporal_feature_selection_list = []
        for x in temporal_feature_selection_model_list:
            # Select relevant features
            temporal_feature_selection = FeatureSelection(
                feature_selection_model_name=x[0],
                feature_type="temporal",
                feature_number=x[1],
                task=task,
                metric_name=metric_name,
                metric_parameters=metric_parameters,
            )
            temporal_feature_selection_list.append(temporal_feature_selection)

        # prediction
        pred_class_list = []

        # Set predictive model
        model_name_list = prediction_model_list

        for model_name in model_name_list:
            # Set model parameters
            model_parameters = {
                "h_dim": 100,
                "n_layer": 2,
                "n_head": 2,
                "batch_size": 128,
                "epoch": 2,
                "model_type": model_name,
                "learning_rate": 0.001,
                "static_mode": "concatenate",
                "time_mode": "concatenate",
                "verbose": False,
            }

            # Train the predictive model
            pred_class = prediction(model_name, model_parameters, task)
            pred_class_list.append(pred_class)

        self.pred_class_list = pred_class_list
        self.temporal_feature_selection_list = temporal_feature_selection_list
        self.static_feature_selection_list = static_feature_selection_list
        self.temporal_imputation_list = temporal_imputation_list
        self.static_imputation_list = static_imputation_list
        self.domain = [
            {"name": "static_imputation", "type": "discrete", "domain": list(range(len(static_imputation_list)))},
            {"name": "temporal_imputation", "type": "discrete", "domain": list(range(len(temporal_imputation_list)))},
            {
                "name": "static_feature_selection",
                "type": "discrete",
                "domain": list(range(len(static_feature_selection_list))),
            },
            {
                "name": "temporal_feature_selection",
                "type": "discrete",
                "domain": list(range(len(temporal_feature_selection_list))),
            },
            {"name": "pred_class", "type": "discrete", "domain": list(range(len(pred_class_list)))},
        ]
        self.myBopt = BayesianOptimization(f=self.f, domain=self.domain)

    def run_opt(self, steps):
        self.myBopt.run_optimization(max_iter=steps)
        opt_sol, opt_obj = self.myBopt.get_evaluations()
        sol = np.where(opt_obj.flatten() == opt_obj.min())
        ind = sol[0]
        best_model = opt_sol[ind]
        best_obj = opt_obj.min()
        best_model = best_model.flatten()
        best_model_list = [
            self.static_imputation_model_list[int(best_model[0])],
            self.temporal_imputation_model_list[int(best_model[1])],
            self.static_feature_selection_model_list[int(best_model[2])],
            self.temporal_feature_selection_model_list[int(best_model[3])],
            self.prediction_model_list[int(best_model[4])],
        ]
        return best_model_list, best_obj

    def f(self, a):
        si, ti, sf, tf, pc = a[0]
        try:
            static_imputation = self.static_imputation_list[int(si)]
            temporal_imputation = self.temporal_imputation_list[int(ti)]
            static_feature_selection = self.static_feature_selection_list[int(sf)]
            temporal_feature_selection = self.temporal_feature_selection_list[int(tf)]
            pred_class = self.pred_class_list[int(pc)]

            pipeline = PipelineComposer(
                static_imputation, temporal_imputation, static_feature_selection, temporal_feature_selection
            )

            dataset_training = pipeline.fit_transform(self.dataset_training)
            dataset_testing = pipeline.transform(self.dataset_testing)

            # only do once
            if not dataset_training.is_validation_defined:
                dataset_training.train_val_test_split(prob_val=0.2, prob_test=0.0)

            # Set up validation for early stopping and best model saving
            pred_class.fit(dataset_training)
            # Return the predictions on the testing set
            test_y_hat = pred_class.predict(dataset_testing)
            metric = BOMetric(metric="auc", fold=0, split="test")
            met_val = metric.eval(dataset_testing, test_y_hat)
            met_val = met_val[met_val != 0].mean()
        except Exception:
            met_val = 1e-9
        return met_val
