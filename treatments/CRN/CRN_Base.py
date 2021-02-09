"""Base class for the Counterfactual Recurrent Network. """

import logging
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper
from tensorflow.python.ops import rnn

from treatments.CRN.flip_gradient import flip_gradient


class CRN_Base:
    def __init__(self, hyperparams, params, b_train_decoder=False, task=None):
        """
        Base class that can be used to initialize the encoder or decoder network as part of the Counterfactual Recurrent
        Network (CRN).

        Args:
            - hyperparams: dictionary with the hyperparameters specifying the architecture of the CRN encoder or decoder model.
            - params: dictionary of parameters specifying the following dimensions needed for initializing the placeholder
                values of the TensorFlow graph: num_treatments (number of treatments), num_covariates (number of covariates),
                num_outputs (number of outputs) and max_sequence_length.
            - b_train_decoder: boolean value indicating whether to train decoder model.
            - task: 'classification' or 'regression'
            - static_mode: 'concatenate' or None
            - time_mode: 'concatenate' or None
        """
        self.br_size = hyperparams["br_size"]
        self.rnn_hidden_units = hyperparams["rnn_hidden_units"]
        self.fc_hidden_units = hyperparams["fc_hidden_units"]
        self.batch_size = hyperparams["batch_size"]
        self.rnn_keep_prob = hyperparams["rnn_keep_prob"]
        self.learning_rate = hyperparams["learning_rate"]
        self.num_epochs = hyperparams["num_epochs"]
        self.max_alpha = hyperparams["max_alpha"]

        self.task = task

        self.b_train_decoder = b_train_decoder
        self.params = params

        if self.b_train_decoder:
            self.name_scope = "decoder"
        else:
            self.name_scope = "encoder"

    def init_model(self):
        """Initialize the placeholder variables in the TensorFlow graph."""

        self.num_treatments = self.params["num_treatments"]
        self.num_covariates = self.params["num_covariates"]
        self.num_outputs = self.params["num_outputs"]
        self.max_sequence_length = self.params["max_sequence_length"]

        self.current_covariates = tf.placeholder(tf.float32, [None, self.max_sequence_length, self.num_covariates])

        # Initial previous treatment needs to consist of zeros (this is done when building the feed dictionary)
        self.previous_treatments = tf.placeholder(tf.float32, [None, self.max_sequence_length, self.num_treatments])
        self.current_treatments = tf.placeholder(tf.float32, [None, self.max_sequence_length, self.num_treatments])
        self.outputs = tf.placeholder(tf.float32, [None, self.max_sequence_length, self.num_outputs])
        self.active_entries = tf.placeholder(tf.float32, [None, self.max_sequence_length, 1])

        self.init_state = None
        if self.b_train_decoder:
            self.init_state = tf.placeholder(tf.float32, [None, self.rnn_hidden_units])

        self.alpha = tf.placeholder(tf.float32, [])  # Gradient reversal scalar

    def build_balancing_representation(self):
        """Process the inputs to the model (history of covariates and previous treatments ) using RNN with LSTM cell to
        build the balancing representation.

        Returns:
            - balancing_representation: balancing representation for each timestep in the sequence.
        """

        self.rnn_input = tf.concat([self.current_covariates, self.previous_treatments], axis=-1)
        self.sequence_length = tf.cast(tf.reduce_sum(tf.reduce_max(self.active_entries, axis=2), axis=1), tf.int32)

        rnn_cell = DropoutWrapper(
            LSTMCell(self.rnn_hidden_units, state_is_tuple=False),
            output_keep_prob=self.rnn_keep_prob,
            state_keep_prob=self.rnn_keep_prob,
            variational_recurrent=True,
            dtype=tf.float32,
        )

        decoder_init_state = None
        if self.b_train_decoder:
            decoder_init_state = tf.concat([self.init_state, self.init_state], axis=-1)

        rnn_output, _ = rnn.dynamic_rnn(
            rnn_cell,
            self.rnn_input,
            initial_state=decoder_init_state,
            dtype=tf.float32,
            sequence_length=self.sequence_length,
        )

        # Flatten to apply same weights to all time steps.
        rnn_output = tf.reshape(rnn_output, [-1, self.rnn_hidden_units])
        balancing_representation = tf.layers.dense(rnn_output, self.br_size, activation=tf.nn.elu)

        return balancing_representation

    def build_treatment_assignments_one_hot(self, balancing_representation):
        """Treatment classifier. Predicts treatment from the built representation.

        Args:
         - balancing_representation: balancing representation for each timestep in the sequence.

        Returns:
            - treatment_prob_predictions: output probabilities for each treatment given the representation.
        """
        balancing_representation_gr = flip_gradient(balancing_representation, self.alpha)

        treatments_network_layer = tf.layers.dense(
            balancing_representation_gr, self.fc_hidden_units, activation=tf.nn.elu
        )
        treatment_logit_predictions = tf.layers.dense(treatments_network_layer, self.num_treatments, activation=None)

        # For multiple treatments
        # treatment_prob_predictions = tf.nn.softmax(treatment_logit_predictions)

        # For binary treatments
        treatment_prob_predictions = tf.nn.sigmoid(treatment_logit_predictions)

        return treatment_prob_predictions

    def build_outcomes(self, balancing_representation):
        """Outcome predictor. Estimates potential outcomes given balancing representation.

        Args:
            - balancing_representation: balancing representation for each timestep in the sequence.

        Return:
            - outcome_predictions: predicted factual outcomes.
        """
        current_treatments_reshape = tf.reshape(self.current_treatments, [-1, self.num_treatments])

        outcome_network_input = tf.concat([balancing_representation, current_treatments_reshape], axis=-1)
        if self.task == "regression":
            outcome_network_layer = tf.layers.dense(outcome_network_input, self.fc_hidden_units, activation=tf.nn.elu)
            outcome_predictions = tf.layers.dense(outcome_network_layer, self.num_outputs, activation=None)
        elif self.task == "classification":
            outcome_network_layer = tf.layers.dense(outcome_network_input, self.fc_hidden_units, activation=tf.nn.tanh)
            outcome_predictions = tf.layers.dense(outcome_network_layer, self.num_outputs, activation=tf.nn.sigmoid)

        return outcome_predictions

    def train(self, dataset_train, dataset_val):
        """Train the CRN encoder or decoder model.

        Args:
            - dataset_train: training dataset.
            - dataset_val: validation dataset.
        """

        with tf.variable_scope(self.name_scope, reuse=tf.AUTO_REUSE):
            self.init_model()
            self.balancing_representation = self.build_balancing_representation()
            self.treatment_prob_predictions = self.build_treatment_assignments_one_hot(self.balancing_representation)
            self.predictions = self.build_outcomes(self.balancing_representation)

            self.loss_treatments = self.compute_loss_treatments(
                target_treatments=self.current_treatments,
                treatment_predictions=self.treatment_prob_predictions,
                active_entries=self.active_entries,
            )

            self.loss_outcomes = self.compute_loss_predictions(self.outputs, self.predictions, self.active_entries)

            self.loss = self.loss_outcomes + self.loss_treatments
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            tf_device = "gpu"
            if tf_device == "cpu":
                tf_config = tf.ConfigProto(log_device_placement=False, device_count={"GPU": 0})
            else:
                tf_config = tf.ConfigProto(log_device_placement=False, device_count={"GPU": 1})
                tf_config.gpu_options.allow_growth = True

            self.sess = tf.Session(config=tf_config)
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())

            for epoch in range(self.num_epochs):
                p = float(epoch) / float(self.num_epochs)
                alpha_current = (2.0 / (1.0 + np.exp(-10.0 * p)) - 1) * self.max_alpha

                iteration = 0
                for (
                    batch_current_covariates,
                    batch_previous_treatments,
                    batch_current_treatments,
                    batch_init_state,
                    batch_outputs,
                    batch_active_entries,
                ) in self.gen_epoch(dataset_train, batch_size=self.batch_size):
                    feed_dict = self.build_feed_dictionary(
                        batch_current_covariates=batch_current_covariates,
                        batch_previous_treatments=batch_previous_treatments,
                        batch_current_treatments=batch_current_treatments,
                        batch_init_state=batch_init_state,
                        batch_active_entries=batch_active_entries,
                        batch_outputs=batch_outputs,
                        alpha_current=alpha_current,
                    )

                    _, pred_curr, training_loss, training_loss_outcomes, training_loss_treatments = self.sess.run(
                        [optimizer, self.predictions, self.loss, self.loss_outcomes, self.loss_treatments],
                        feed_dict=feed_dict,
                    )
                    iteration += 1

                # Training loss
                logging.info(
                    "Epoch {} | total loss = {} | outcome loss = {} | "
                    "treatment loss = {} | current alpha = {} ".format(
                        epoch, training_loss, training_loss_outcomes, training_loss_treatments, alpha_current
                    )
                )

                # Validation loss
                validation_loss, validation_loss_outcomes, validation_loss_treatments = self.compute_validation_loss(
                    dataset_val
                )
                logging.info(
                    "Epoch {} Summary| Validation loss = {} | Validation loss outcomes = {} | Validation loss treatments = {}".format(
                        epoch, validation_loss, validation_loss_outcomes, validation_loss_treatments
                    )
                )

    def load_model(self, model_name, model_folder):
        """Load trained CRN model.

        Args:
            - model_name: name of saved model.
            - model_folder: directory with saved model.
        """
        with tf.variable_scope(self.name_scope, reuse=tf.AUTO_REUSE):
            self.init_model()
            self.balancing_representation = self.build_balancing_representation()
            self.treatment_prob_predictions = self.build_treatment_assignments_one_hot(self.balancing_representation)
            self.predictions = self.build_outcomes(self.balancing_representation)

            tf_device = "gpu"
            if tf_device == "cpu":
                tf_config = tf.ConfigProto(log_device_placement=False, device_count={"GPU": 0})
            else:
                tf_config = tf.ConfigProto(log_device_placement=False, device_count={"GPU": 1})
                tf_config.gpu_options.allow_growth = True

            self.sess = tf.Session(config=tf_config)
            self.sess.run(tf.global_variables_initializer())
            checkpoint_name = model_name
            self.load_network(self.sess, model_folder, checkpoint_name)

    def build_feed_dictionary(
        self,
        batch_current_covariates,
        batch_previous_treatments,
        batch_current_treatments,
        batch_init_state,
        batch_active_entries,
        batch_outputs=None,
        alpha_current=1.0,
        training_mode=True,
    ):
        """Build feed dictionary for TensorFlow graph."""

        batch_size = batch_previous_treatments.shape[0]
        zero_init_treatment = np.zeros(shape=[batch_size, 1, self.num_treatments])
        new_batch_previous_treatments = np.concatenate([zero_init_treatment, batch_previous_treatments], axis=1)

        if training_mode:
            if self.b_train_decoder:
                feed_dict = {
                    self.current_covariates: batch_current_covariates,
                    self.previous_treatments: batch_previous_treatments,
                    self.current_treatments: batch_current_treatments,
                    self.init_state: batch_init_state,
                    self.outputs: batch_outputs,
                    self.active_entries: batch_active_entries,
                    self.alpha: alpha_current,
                }

            else:
                feed_dict = {
                    self.current_covariates: batch_current_covariates,
                    self.previous_treatments: new_batch_previous_treatments,
                    self.current_treatments: batch_current_treatments,
                    self.outputs: batch_outputs,
                    self.active_entries: batch_active_entries,
                    self.alpha: alpha_current,
                }
        else:
            if self.b_train_decoder:
                feed_dict = {
                    self.current_covariates: batch_current_covariates,
                    self.previous_treatments: batch_previous_treatments,
                    self.current_treatments: batch_current_treatments,
                    self.init_state: batch_init_state,
                    self.active_entries: batch_active_entries,
                    self.alpha: alpha_current,
                }
            else:
                feed_dict = {
                    self.current_covariates: batch_current_covariates,
                    self.previous_treatments: new_batch_previous_treatments,
                    self.current_treatments: batch_current_treatments,
                    self.active_entries: batch_active_entries,
                    self.alpha: alpha_current,
                }

        return feed_dict

    def gen_epoch(self, dataset, batch_size, training_mode=True):
        """Generates epoch from training, by splitting the dataset into chunks of batch size."""

        dataset_size = dataset["current_covariates"].shape[0]
        num_batches = int(dataset_size / batch_size) + 1

        for i in range(num_batches):
            if i == num_batches - 1:
                batch_samples = range(dataset_size - batch_size, dataset_size)
            else:
                batch_samples = range(i * batch_size, (i + 1) * batch_size)

            if training_mode:
                batch_current_covariates = dataset["current_covariates"][batch_samples, :, :]
                batch_previous_treatments = dataset["previous_treatments"][batch_samples, :, :]
                batch_current_treatments = dataset["current_treatments"][batch_samples, :, :]
                batch_outputs = dataset["outputs"][batch_samples, :, :]
                batch_active_entries = dataset["active_entries"][batch_samples, :, :]

                batch_init_state = None
                if self.b_train_decoder:
                    batch_init_state = dataset["init_state"][batch_samples, :]

                yield (
                    batch_current_covariates,
                    batch_previous_treatments,
                    batch_current_treatments,
                    batch_init_state,
                    batch_outputs,
                    batch_active_entries,
                )
            else:
                batch_current_covariates = dataset["current_covariates"][batch_samples, :, :]
                batch_previous_treatments = dataset["previous_treatments"][batch_samples, :, :]
                batch_current_treatments = dataset["current_treatments"][batch_samples, :, :]
                batch_active_entries = dataset["active_entries"][batch_samples, :, :]

                batch_init_state = None
                if self.b_train_decoder:
                    batch_init_state = dataset["init_state"][batch_samples, :]

                yield (
                    batch_current_covariates,
                    batch_previous_treatments,
                    batch_current_treatments,
                    batch_init_state,
                    batch_active_entries,
                )

    def get_balancing_reps(self, dataset):
        """Compute balancing representations for patients in the dataset.

        Args:
            - dataset: dataset with test patients
        Returns:
            - balancing representations at each timestep for patients in the dataset.
        """

        dataset_size = dataset["current_covariates"].shape[0]
        balancing_reps = np.zeros(shape=(dataset_size, self.max_sequence_length, self.br_size))

        dataset_size = dataset["current_covariates"].shape[0]
        if dataset_size > 10000:  # Does not fit into memory
            batch_size = 10000
        else:
            batch_size = dataset_size

        num_batches = int(dataset_size / batch_size) + 1

        batch_id = 0
        num_samples = 1
        for (
            batch_current_covariates,
            batch_previous_treatments,
            batch_current_treatments,
            batch_init_state,
            batch_active_entries,
        ) in self.gen_epoch(dataset, batch_size=batch_size, training_mode=False):
            feed_dict = self.build_feed_dictionary(
                batch_current_covariates=batch_current_covariates,
                batch_previous_treatments=batch_previous_treatments,
                batch_current_treatments=batch_current_treatments,
                batch_active_entries=batch_active_entries,
                batch_init_state=batch_init_state,
                training_mode=False,
            )

            # Dropout samples
            total_predictions = np.zeros(shape=(batch_size, self.max_sequence_length, self.br_size))

            for sample in range(num_samples):
                br_outputs = self.sess.run(self.balancing_representation, feed_dict=feed_dict)
                br_outputs = np.reshape(br_outputs, newshape=(-1, self.max_sequence_length, self.br_size))
                total_predictions += br_outputs

            total_predictions /= num_samples

            if batch_id == num_batches - 1:
                batch_samples = range(dataset_size - batch_size, dataset_size)
            else:
                batch_samples = range(batch_id * batch_size, (batch_id + 1) * batch_size)

            batch_id += 1
            balancing_reps[batch_samples] = total_predictions

        return balancing_reps

    def get_predictions(self, dataset, b_with_uncertainty=False):
        """Estimate one-step-ahead patient outcomes.

        Args:
            - dataset: dataset with test patients.
            - b_with_uncertainty: boolean indicating whether to return uncertainty estimates for the predictions

        Returns:
            - predictions: predicted one-step-ahead outcomes for patients in the dataset.
        """
        dataset_size = dataset["current_covariates"].shape[0]

        predictions_mean = np.zeros(shape=(dataset_size, self.max_sequence_length, self.num_outputs))
        predictions_std = np.zeros(shape=(dataset_size, self.max_sequence_length, self.num_outputs))

        dataset_size = dataset["current_covariates"].shape[0]
        if dataset_size > 10000:
            batch_size = 10000
        else:
            batch_size = dataset_size

        num_batches = int(dataset_size / batch_size) + 1

        batch_id = 0
        num_samples = 1
        for (
            batch_current_covariates,
            batch_previous_treatments,
            batch_current_treatments,
            batch_init_state,
            batch_active_entries,
        ) in self.gen_epoch(dataset, batch_size=batch_size, training_mode=False):
            feed_dict = self.build_feed_dictionary(
                batch_current_covariates=batch_current_covariates,
                batch_previous_treatments=batch_previous_treatments,
                batch_current_treatments=batch_current_treatments,
                batch_init_state=batch_init_state,
                batch_active_entries=batch_active_entries,
                training_mode=False,
            )

            # Dropout samples
            total_predictions = []

            for sample in range(num_samples):
                predicted_outputs = self.sess.run(self.predictions, feed_dict=feed_dict)
                predicted_outputs = np.reshape(
                    predicted_outputs, newshape=(-1, self.max_sequence_length, self.num_outputs)
                )

                total_predictions.append(predicted_outputs)

            total_predictions = np.array(total_predictions)

            if batch_id == num_batches - 1:
                batch_samples = range(dataset_size - batch_size, dataset_size)
            else:
                batch_samples = range(batch_id * batch_size, (batch_id + 1) * batch_size)

            batch_id += 1
            predictions_mean[batch_samples] = np.mean(total_predictions, axis=0)
            predictions_std[batch_samples] = np.std(total_predictions, axis=0)

        predictions = dict()
        predictions["mean"] = predictions_mean
        predictions["std"] = predictions_std

        if not b_with_uncertainty:
            predictions = predictions["mean"]

        return predictions

    def get_autoregressive_sequence_predictions(self, test_data, b_with_uncertainty=False):
        """Estimate patient outcomes under a sequence of future treatments given their past history of treatment and
             covariates. Can only be used for the decoder model.

        Args:
            - test_data: dataset dictionary consisting of sequences of future treatments for which outcomes are estimated.
            - b_with_uncertainty: boolean indicating whether to return uncertainty estimates for the predictions

        Returns:
            - predicted_outputs: counterfactual outcomes for a sequence of future treatments.
        """

        encoder_output = test_data["encoder_output"]
        current_treatments = test_data["current_treatments"]
        previous_treatments = test_data["previous_treatments"]
        init_states = test_data["init_states"]

        projection_horizon = current_treatments.shape[1]

        num_patient_points = encoder_output.shape[0]

        current_dataset = dict()
        current_dataset["current_covariates"] = np.zeros(
            shape=(num_patient_points, projection_horizon, encoder_output.shape[-1])
        )
        current_dataset["previous_treatments"] = np.zeros(
            shape=(num_patient_points, projection_horizon, self.num_treatments)
        )
        current_dataset["current_treatments"] = np.zeros(
            shape=(num_patient_points, projection_horizon, self.num_treatments)
        )
        current_dataset["active_entries"] = np.ones(shape=(num_patient_points, projection_horizon, 1))
        current_dataset["init_state"] = np.zeros((num_patient_points, init_states.shape[-1]))

        predicted_outputs = dict()
        predicted_outputs["mean"] = np.zeros(shape=(num_patient_points, projection_horizon, self.num_outputs))
        predicted_outputs["std"] = np.zeros(shape=(num_patient_points, projection_horizon, self.num_outputs))

        for i in range(num_patient_points):
            current_dataset["init_state"][i] = init_states[i]
            current_dataset["current_covariates"][i, 0] = encoder_output[i]
            current_dataset["previous_treatments"][i] = previous_treatments[i]
            current_dataset["current_treatments"][i] = current_treatments[i]

        for t in range(0, projection_horizon):
            predictions = self.get_predictions(current_dataset, b_with_uncertainty=True)
            for i in range(num_patient_points):
                predicted_outputs["mean"][i, t] = predictions["mean"][i, t]
                predicted_outputs["std"][i, t] = predictions["std"][i, t]
                if t < projection_horizon - 1:
                    current_dataset["current_covariates"][i, t + 1, 0] = predictions["mean"][i, t, 0]

        test_data["predicted_outcomes"] = predicted_outputs

        if not b_with_uncertainty:
            predicted_outputs = predicted_outputs["mean"]

        return predicted_outputs

    def compute_loss_treatments(self, target_treatments, treatment_predictions, active_entries):
        """Computes binary cross-entropy loss for treatment prediction."""
        treatment_predictions = tf.reshape(treatment_predictions, [-1, self.max_sequence_length, self.num_treatments])

        cross_entropy_loss = tf.reduce_sum(
            (
                -(
                    target_treatments * tf.log(treatment_predictions + 1e-8)
                    + (1 - target_treatments) * tf.log(1 - treatment_predictions + 1e-8)
                )
            )
            * active_entries
        ) / tf.reduce_sum(active_entries)

        return cross_entropy_loss

    def compute_loss_predictions(self, outputs, predictions, active_entries):
        """Computes loss for outcome prediction. The loss is mean sqa"""
        predictions = tf.reshape(predictions, [-1, self.max_sequence_length, self.num_outputs])

        if self.task == "regression":
            loss = tf.reduce_sum(tf.square(outputs - predictions) * active_entries) / tf.reduce_sum(active_entries)
        elif self.task == "classification":
            loss = tf.reduce_sum(
                (-(outputs * tf.log(predictions + 1e-8) + (1 - outputs) * tf.log(1 - predictions + 1e-8)))
                * active_entries
            ) / tf.reduce_sum(active_entries)
        else:
            raise Exception("Unknown task")

        return loss

    def compute_validation_loss(self, validation_dataset):
        """Computes validation losses for the validation dataset."""
        validation_losses = []
        validation_losses_outcomes = []
        validation_losses_treatments = []

        dataset_size = validation_dataset["current_covariates"].shape[0]
        if dataset_size > 10000:
            batch_size = 10000
        else:
            batch_size = dataset_size

        for (
            batch_current_covariates,
            batch_previous_treatments,
            batch_current_treatments,
            batch_init_state,
            batch_outputs,
            batch_active_entries,
        ) in self.gen_epoch(validation_dataset, batch_size=batch_size):
            feed_dict = self.build_feed_dictionary(
                batch_current_covariates=batch_current_covariates,
                batch_previous_treatments=batch_previous_treatments,
                batch_current_treatments=batch_current_treatments,
                batch_init_state=batch_init_state,
                batch_active_entries=batch_active_entries,
                batch_outputs=batch_outputs,
            )

            validation_loss, validation_loss_outcomes, validation_loss_treatments = self.sess.run(
                [self.loss, self.loss_outcomes, self.loss_treatments], feed_dict=feed_dict
            )

            validation_losses.append(validation_loss)
            validation_losses_outcomes.append(validation_loss_outcomes)
            validation_losses_treatments.append(validation_loss_treatments)

        validation_loss = np.mean(np.array(validation_losses))
        validation_loss_outcomes = np.mean(np.array(validation_losses_outcomes))
        validation_loss_treatments = np.mean(np.array(validation_losses_treatments))

        return validation_loss, validation_loss_outcomes, validation_loss_treatments

    def save_network(self, model_dir, checkpoint_name):
        """Save trained network.

        Args:
            - model_dir: directory where to save the model.
            - checkpoint_name: saved model name.
        """
        varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_scope)
        saver = tf.train.Saver(var_list=varlist, max_to_keep=100000)

        save_path = saver.save(self.sess, os.path.join(model_dir, "{0}.ckpt".format(checkpoint_name)))
        logging.info("Model saved to: {0}".format(save_path))

    def load_network(self, tf_session, model_dir, checkpoint_name):
        """Load trained network into a TensorFlow session.

        Args:
            - tf_session: TensorFlow session.
            - model_dir: directory with saved model.
            - checkpoint_name: saved model name.
        """
        load_path = os.path.join(model_dir, "{0}.ckpt".format(checkpoint_name))
        logging.info("Restoring model from {0}".format(load_path))

        varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_scope)
        saver = tf.train.Saver(var_list=varlist)
        saver.restore(tf_session, load_path)
