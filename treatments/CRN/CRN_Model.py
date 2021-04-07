"""Counterfactual Recurrent Network. Treatment effects model."""

import numpy as np
import tensorflow as tf
import pickle
import os
from treatments.CRN.data_utils import data_preprocess, data_preprocess_counterfactuals, process_seq_data
from treatments.CRN.CRN_Base import CRN_Base

from base import BaseEstimator, PredictorMixin


class CRN_Model(BaseEstimator, PredictorMixin):
    def __init__(
        self,
        encoder_rnn_hidden_units=None,
        encoder_br_size=None,
        encoder_fc_hidden_units=None,
        encoder_learning_rate=None,
        encoder_batch_size=None,
        encoder_keep_prob=None,
        encoder_num_epochs=None,
        encoder_max_alpha=None,
        decoder_br_size=None,
        decoder_fc_hidden_units=None,
        decoder_learning_rate=None,
        decoder_batch_size=None,
        decoder_keep_prob=None,
        decoder_num_epochs=None,
        decoder_max_alpha=None,
        projection_horizon=None,
        task=None,
        static_mode=None,
        time_mode=None,
        model_id="crn_model",
    ):
        """
        Initialize the Counterfactual Recurrent Network (CRN).

        Args:
            - hyperparams_encoder: dictionary with the hyperparameters specifying the architecture of the CRN encoder model.
            - hyperparams_decoder: dictionary with the hyperparameters specifying the architecture of the CRN decoder model.
            - task: 'classification' or 'regression'
            - static_mode: 'concatenate' or None
            - time_mode: 'concatenate' or None
        """
        super().__init__(task)

        self.encoder_rnn_hidden_units = encoder_rnn_hidden_units
        self.encoder_br_size = encoder_br_size
        self.encoder_fc_hidden_units = encoder_fc_hidden_units
        self.encoder_learning_rate = encoder_learning_rate
        self.encoder_batch_size = encoder_batch_size
        self.encoder_keep_prob = encoder_keep_prob
        self.encoder_num_epochs = encoder_num_epochs
        self.encoder_max_alpha = encoder_max_alpha

        self.decoder_br_size = decoder_br_size
        self.decoder_fc_hidden_units = decoder_fc_hidden_units
        self.decoder_learning_rate = decoder_learning_rate
        self.decoder_batch_size = decoder_batch_size
        self.decoder_keep_prob = decoder_keep_prob
        self.decoder_num_epochs = decoder_num_epochs
        self.decoder_max_alpha = decoder_max_alpha

        self.projection_horizon = projection_horizon
        self.task = task
        self.static_mode = static_mode
        self.time_mode = time_mode
        self.model_id = model_id

    def new(self, model_id):
        """Create a new model with the same parameter as the existing one.

        Args:
            - model_id: an unique identifier for the new model
        Returns:
            - a new CRN_Model
        """
        return CRN_Model(
            encoder_rnn_hidden_units=self.encoder_rnn_hidden_units,
            encoder_br_size=self.encoder_br_size,
            encoder_fc_hidden_units=self.encoder_fc_hidden_units,
            encoder_learning_rate=self.encoder_learning_rate,
            encoder_batch_size=self.encoder_batch_size,
            encoder_keep_prob=self.encoder_keep_prob,
            encoder_num_epochs=self.encoder_num_epochs,
            encoder_max_alpha=self.encoder_max_alpha,
            decoder_br_size=self.decoder_br_size,
            decoder_fc_hidden_units=self.decoder_fc_hidden_units,
            decoder_learning_rate=self.decoder_learning_rate,
            decoder_batch_size=self.decoder_batch_size,
            decoder_keep_prob=self.decoder_keep_prob,
            decoder_num_epochs=self.decoder_num_epochs,
            decoder_max_alpha=self.decoder_max_alpha,
            task=self.task,
            static_mode=self.static_mode,
            time_mode=self.time_mode,
            model_id=model_id,
        )

    @staticmethod
    def get_hyperparameter_space():
        hyp_ = [
            {"name": "encoder_rnn_hidden_units", "type": "discrete", "domain": [64, 128, 256], "dimensionality": 1},
            {"name": "encoder_br_size", "type": "discrete", "domain": [32, 64, 128], "dimensionality": 1},
            {"name": "encoder_fc_hidden_units", "type": "discrete", "domain": [32, 64, 128], "dimensionality": 1},
            {"name": "encoder_learning_rate", "type": "continuous", "domain": [0.001, 0.01], "dimensionality": 1},
            {"name": "encoder_batch_size", "type": "discrete", "domain": [128, 256, 512], "dimensionality": 1},
            {"name": "encoder_keep_prob", "type": "continuous", "domain": [0.7, 0.9], "dimensionality": 1},
            {"name": "encoder_num_epochs", "type": "discrete", "domain": [100], "dimensionality": 1},
            {"name": "encoder_max_alpha", "type": "continuous", "domain": [0.1, 1.0], "dimensionality": 1},
            {"name": "decoder_br_size", "type": "discrete", "domain": [32, 64, 128], "dimensionality": 1},
            {"name": "decoder_fc_hidden_units", "type": "discrete", "domain": [32, 64, 128], "dimensionality": 1},
            {"name": "decoder_learning_rate", "type": "continuous", "domain": [0.001, 0.01], "dimensionality": 1},
            {"name": "decoder_batch_size", "type": "discrete", "domain": [256, 512, 1024], "dimensionality": 1},
            {"name": "decoder_keep_prob", "type": "continuous", "domain": [0.7, 0.9], "dimensionality": 1},
            {"name": "decoder_num_epochs", "type": "discrete", "domain": [100], "dimensionality": 1},
            {"name": "decoder_max_alpha", "type": "continuous", "domain": [0.1, 1.0], "dimensionality": 1},
        ]
        return hyp_

    def fit(self, dataset, fold=0, train_split="train", val_split="val"):
        """Fit the treatment effects encoder model model.

        Args:
            - dataset: temporal, static, label, time, treatment information
            - projection_horizon: number of future timesteps to use for training decoder model.≈¬
            - fold: Cross validation fold
            - train_split: training set splitting parameter
            - valid_split: validation set splitting parameter
        """
        dataset_crn_train = data_preprocess(dataset, fold, train_split, self.static_mode, self.time_mode)
        dataset_crn_val = data_preprocess(dataset, fold, val_split, self.static_mode, self.time_mode)

        num_outputs = dataset_crn_train["outputs"].shape[-1]
        max_sequence_length = dataset_crn_train["current_covariates"].shape[1]
        num_covariates = dataset_crn_train["current_covariates"].shape[-1]

        self.hyperparams_encoder = {
            "rnn_hidden_units": self.encoder_rnn_hidden_units,
            "br_size": self.encoder_br_size,
            "fc_hidden_units": self.encoder_fc_hidden_units,
            "learning_rate": self.encoder_learning_rate,
            "batch_size": self.encoder_batch_size,
            "rnn_keep_prob": self.encoder_keep_prob,
            "num_epochs": self.encoder_num_epochs,
            "max_alpha": self.encoder_max_alpha,
        }

        self.hyperparams_decoder = {
            "rnn_hidden_units": self.hyperparams_encoder["br_size"],
            "br_size": self.decoder_br_size,
            "fc_hidden_units": self.decoder_fc_hidden_units,
            "learning_rate": self.decoder_learning_rate,
            "batch_size": self.decoder_batch_size,
            "rnn_keep_prob": self.decoder_keep_prob,
            "num_epochs": self.decoder_num_epochs,
            "max_alpha": self.decoder_max_alpha,
        }

        with tf.variable_scope(self.model_id, reuse=tf.AUTO_REUSE):
            self.encoder_params = {
                "num_treatments": 2,
                "num_covariates": num_covariates,
                "num_outputs": num_outputs,
                "max_sequence_length": max_sequence_length,
            }

            self.encoder_model = CRN_Base(self.hyperparams_encoder, self.encoder_params, task=self.task)
            self.encoder_model.train(dataset_crn_train, dataset_crn_val)

            if self.projection_horizon is not None:
                training_br_states = self.encoder_model.get_balancing_reps(dataset_crn_train)
                validation_br_states = self.encoder_model.get_balancing_reps(dataset_crn_val)

                training_seq_processed = process_seq_data(
                    dataset_crn_train, training_br_states, self.projection_horizon
                )
                validation_seq_processed = process_seq_data(
                    dataset_crn_val, validation_br_states, self.projection_horizon
                )

                num_outputs = training_seq_processed["outputs"].shape[-1]
                num_covariates = training_seq_processed["current_covariates"].shape[-1]

                self.decoder_params = {
                    "num_treatments": 2,
                    "num_covariates": num_covariates,
                    "num_outputs": num_outputs,
                    "max_sequence_length": self.projection_horizon,
                }

                self.decoder_model = CRN_Base(
                    self.hyperparams_decoder, self.decoder_params, b_train_decoder=True, task=self.task
                )
                self.decoder_model.train(training_seq_processed, validation_seq_processed)

    def predict(self, dataset, fold=0, test_split="test"):
        """Return the one-step-ahead predicted outcomes on the test set. These are one-step-ahead predictions.

        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Test fold
            - test_split: testing set splitting parameter

        Returns:
            - test_y_hat: predictions on testing set
        """
        with tf.variable_scope(self.model_id, reuse=tf.AUTO_REUSE):
            dataset_crn_test = data_preprocess(dataset, fold, test_split, self.static_mode, self.time_mode)
            test_y_hat = self.encoder_model.get_predictions(dataset_crn_test)
            return test_y_hat

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

        with tf.variable_scope(self.model_id, reuse=tf.AUTO_REUSE):
            history, encoder_output, dataset_crn_decoder = data_preprocess_counterfactuals(
                encoder_model=self.encoder_model,
                dataset=dataset,
                patient_id=patient_id,
                timestep=timestep,
                treatment_options=treatment_options,
                fold=fold,
                split=test_split,
                static_mode=self.static_mode,
                time_mode=self.time_mode,
            )
            decoder_outputs = self.decoder_model.get_autoregressive_sequence_predictions(dataset_crn_decoder)
            counterfactual_trajectories = np.concatenate([encoder_output, decoder_outputs], axis=1)

        return history, counterfactual_trajectories

    def save_model(self, model_dir, model_name):
        """Save the model to model_dir using the model_name.

        Args:
            - model_dir: directory where to save the model
            - model_name: name of saved model
        """
        encoder_model_name = "encoder_" + model_name
        self.encoder_model.save_network(model_dir, encoder_model_name, DEBUG_scope_prefix="crn_model/")
        pickle.dump(self.encoder_params, open(os.path.join(model_dir, "encoder_params_" + model_name + ".pkl"), "wb"))
        pickle.dump(
            self.hyperparams_encoder, open(os.path.join(model_dir, "hyperparams_encoder_" + model_name + ".pkl"), "wb")
        )

        decoder_model_name = "decoder_" + model_name
        self.decoder_model.save_network(model_dir, decoder_model_name, DEBUG_scope_prefix="crn_model/")
        pickle.dump(self.decoder_params, open(os.path.join(model_dir, "decoder_params_" + model_name + ".pkl"), "wb"))
        pickle.dump(
            self.hyperparams_decoder, open(os.path.join(model_dir, "hyperparams_decoder_" + model_name + ".pkl"), "wb")
        )

    def load_model(self, model_dir, model_name):
        """
        Load and return the model from model_path

        Args:
            - model_path:    dictionary containing model_dir (directory where to save the model) and model_name for the
                                         the saved encoder and decoder models
        """
        encoder_params = pickle.load(open(os.path.join(model_dir, "encoder_params_" + model_name + ".pkl"), "rb"))
        encoder_hyperparams = pickle.load(
            open(os.path.join(model_dir, "hyperparams_encoder_" + model_name + ".pkl"), "rb")
        )
        encoder_model_name = "encoder_" + model_name

        encoder_model = CRN_Base(encoder_hyperparams, encoder_params, task=self.task)
        encoder_model.load_model(model_name=encoder_model_name, model_folder=model_dir, DEBUG_scope_prefix="crn_model/")

        decoder_params = pickle.load(open(os.path.join(model_dir, "decoder_params_" + model_name + ".pkl"), "rb"))
        decoder_hyperparams = pickle.load(
            open(os.path.join(model_dir, "hyperparams_decoder_" + model_name + ".pkl"), "rb")
        )
        decoder_model_name = "decoder_" + model_name
        decoder_model = CRN_Base(decoder_hyperparams, decoder_params, b_train_decoder=True, task=self.task)
        decoder_model.load_model(model_name=decoder_model_name, model_folder=model_dir, DEBUG_scope_prefix="crn_model/")

        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
