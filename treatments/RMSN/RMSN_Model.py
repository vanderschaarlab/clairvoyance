"""Recurrent Marginal Structural Networks. Treatment effects model."""

import numpy as np

import os

from treatments.RMSN.script_rnn_fit import rnn_fit
from treatments.RMSN.script_propensity_generation import propensity_generation
from treatments.RMSN.script_rnn_test import rnn_test
from treatments.RMSN.script_decoder_fit import decoder_fit
from treatments.RMSN.script_decoder_test import decoder_test

from base import BaseEstimator, PredictorMixin
from utils.data_utils import concate_xs, concate_xt


class RMSN_Model(BaseEstimator, PredictorMixin):
    def __init__(
        self,
        hyperparams_encoder_iptw=None,
        hyperparams_decoder_iptw=None,
        model_dir=None,
        model_name=None,
        task=None,
        static_mode=None,
        time_mode=None,
    ):
        """
        Initialize the Recurrent Marginal Structural Networks (RMSNs).

        Args:
            - hyperparams_encoder_iptw: hyperparameters for the propensity weighted encoder
            - hyperparams_decoder_iptw: hyperparameters for the propensity weighted decoder
            - model_dir: directory where to save the model
            - model_name: model name
            - task: 'classification' or 'regression'
            - static_mode: 'concatenate' or None
            - time_mode: 'concatenate' or None
        """
        super().__init__(task)
        self.static_mode = static_mode
        self.time_mode = time_mode

        self.hyperparams_encoder_iptw = hyperparams_encoder_iptw
        self.hyperparams_decoder_iptw = hyperparams_decoder_iptw

        self.model_dir = model_dir
        self.model_name = model_name

        self.task = task

    def data_preprocess(self, dataset, fold, split):
        """Preprocess the dataset.

        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Cross validation fold
            - split: 'train', 'valid' or 'test'

        Returns:
            - dataset: dataset dictionary for training the RMSN.
        """
        x, s, y, t, treat = dataset.get_fold(fold, split)

        if self.static_mode == "concatenate":
            x = concate_xs(x, s)

        if self.time_mode == "concatenate":
            x = concate_xt(x, t)

        dataset = dict()
        treat = np.round(treat)

        active_entries = np.ndarray.max((y >= 0).astype(float), axis=-1)
        sequence_lengths = np.sum(active_entries, axis=1).astype(int)

        active_entries = active_entries[:, :, np.newaxis]

        dataset["current_covariates"] = x
        dataset["current_treatments"] = treat
        dataset["previous_treatments"] = np.concatenate(
            [np.zeros(shape=(treat.shape[0], 1, treat.shape[-1])), treat[:, :-1, :]], axis=1
        )
        dataset["outputs"] = y
        dataset["active_entries"] = active_entries
        dataset["sequence_lengths"] = sequence_lengths

        return dataset

    def data_preprocess_counterfactuals(
        self, dataset, patient_id, timestep, treatment_options, fold, split, static_mode, time_mode
    ):
        """Preprocess the dataset for obtaining counterfactual predictions for sequences of future treatments.

        Args:
            - dataset: temporal, static, label, time, treatment information
            - patient_id: patient id of patient for which the counterfactuals are computed
            - timestep: timestep in the patient trajectory where counterfactuals are predicted
            - treatment_options: treatment options for computing the counterfactual trajectories
            - fold: test fold
            - test_split: testing set splitting parameter
            - static_mode: 'concatenate' or None
            - time_mode: 'concatenate' or None

        Returns:
            - patient_history: history of patient outcome until the specified timestep
            - encoder_output: patient output for the first treatment in the treatment options; this one-step-ahead prediction
                is made using the encoder model.
            - dataset_decoder: dataset that can be used to obtain the counterfactual predictions from the decoder model.

        """
        x, s, y, t, treat = dataset.get_fold(fold, split)

        max_sequence_length = x.shape[1]
        num_treatment_options = treatment_options.shape[0]
        projection_horizon = treatment_options.shape[1] - 1

        if static_mode == "concatenate":
            x = concate_xs(x, s)

        if time_mode == "concatenate":
            x = concate_xt(x, t)

        x = np.repeat([x[patient_id]], num_treatment_options, axis=0)
        y = np.repeat([y[patient_id]], num_treatment_options, axis=0)
        treat = np.repeat([treat[patient_id][: timestep - 1]], num_treatment_options, axis=0)
        treat = np.concatenate([treat, treatment_options], axis=1)

        dataset_encoder = dict()

        treatments_encoder = treat[:, :timestep, :]
        treatments_encoder = np.concatenate(
            [treatments_encoder, np.zeros(shape=(treat.shape[0], max_sequence_length - timestep, treat.shape[-1]))],
            axis=1,
        )

        dataset_encoder["current_covariates"] = x
        dataset_encoder["current_treatments"] = treatments_encoder
        dataset_encoder["previous_treatments"] = np.concatenate(
            [np.zeros(shape=(treat.shape[0], 1, treatments_encoder.shape[-1])), treatments_encoder[:, :-1, :]], axis=1
        )
        dataset_encoder["outputs"] = y
        dataset_encoder["active_entries"] = np.ones(shape=(x.shape[0], x.shape[1], 1))
        dataset_encoder["sequence_lengths"] = timestep * np.ones(shape=(num_treatment_options))

        test_encoder_predictions, test_states = rnn_test(dataset_encoder, self.task, self.MODEL_ROOT)

        treatments_decoder = treat[:, timestep : timestep + projection_horizon, :]

        dataset_decoder = dict()
        dataset_decoder["initial_states"] = test_states[:, timestep - 1, :]
        dataset_decoder["scaled_inputs"] = treatments_decoder
        dataset_decoder["scaled_outputs"] = np.zeros(shape=(y.shape[0], projection_horizon, y.shape[-1]))
        dataset_decoder["active_entries"] = treatments_decoder
        dataset_decoder["sequence_lengths"] = projection_horizon * np.ones(shape=(num_treatment_options))

        patient_history = y[0][:timestep]
        encoder_output = test_encoder_predictions[:, timestep - 1 : timestep, :]

        return patient_history, encoder_output, dataset_decoder

    def fit(self, dataset, projection_horizon=None, fold=0, train_split="train", val_split="val"):
        """Fit the treatment effects model model.

        Args:
            - dataset: temporal, static, label, time, treatment information
            - projection_horizon: number of future timesteps for training decoder
            - fold: Cross validation fold
            - train_split: training set splitting parameter
            - valid_split: validation set splitting parameter
        """
        dataset_train = self.data_preprocess(dataset, fold, train_split)
        dataset_val = self.data_preprocess(dataset, fold, val_split)

        self.MODEL_ROOT = os.path.join(self.model_dir, self.model_name)

        if not os.path.exists(self.MODEL_ROOT):
            os.makedirs(self.MODEL_ROOT)

        # Train the model
        rnn_fit(
            self.hyperparams_encoder_iptw,
            dataset_train,
            dataset_val,
            networks_to_train="propensity_networks",
            task=self.task,
            MODEL_ROOT=self.MODEL_ROOT,
        )
        propensity_generation(dataset_train, dataset_val, MODEL_ROOT=self.MODEL_ROOT)
        rnn_fit(
            self.hyperparams_encoder_iptw,
            dataset_train,
            dataset_val,
            networks_to_train="encoder",
            task=self.task,
            MODEL_ROOT=self.MODEL_ROOT,
        )

        if projection_horizon is not None:
            decoder_fit(
                self.hyperparams_decoder_iptw,
                dataset_train,
                dataset_val,
                projection_horizon=projection_horizon,
                task=self.task,
                MODEL_ROOT=self.MODEL_ROOT,
            )

    def predict_counterfactual_trajectories(
        self, dataset, patient_id, timestep, treatment_options, fold=0, test_split="test"
    ):
        """Return the counterfactual trajectories for a patient and for the specified future treatment options.

        Args:
            - dataset: dataset with test patients
            - patient_id: patient id of patient for which the counterfactuals are computed
            - timestep: timestep in the patient trajectory where counterfactuals are predicted
            - treatment_options: treatment options for computing the counterfactual trajectories; the length of the
                sequence of treatment options needs to be projection_horizon + 1 where projection_horizon is the number of
                future timesteps used for training decoder model.
            - fold: test fold
            - test_split: testing set splitting parameter

        Returns:
            - history: history of previous outputs for the patient.
            - counterfactual_trajectories: trajectories of counterfactual predictions for the specified future treatments
                in the treatment_options
        """

        history, encoder_output, dataset_decoder = self.data_preprocess_counterfactuals(
            dataset=dataset,
            patient_id=patient_id,
            timestep=timestep,
            treatment_options=treatment_options,
            fold=fold,
            split=test_split,
            static_mode=self.static_mode,
            time_mode=self.time_mode,
        )
        decoder_outputs = decoder_test(dataset_decoder, self.task, self.MODEL_ROOT)
        counterfactual_trajectories = np.concatenate([encoder_output, decoder_outputs], axis=1)

        return history, counterfactual_trajectories

    def predict(self, dataset, fold=0, test_split="test"):
        """Return the predicted factual outcomes on the test set. These are one-step-ahead predictions.

        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Cross validation fold
            - test_split: testing set splitting parameter

        Returns:
            - test_y_hat: predictions on testing set
        """
        dataset_test = self.data_preprocess(dataset, fold, test_split)
        test_y_hat, _ = rnn_test(dataset_test, self.task, self.MODEL_ROOT)
        return test_y_hat
