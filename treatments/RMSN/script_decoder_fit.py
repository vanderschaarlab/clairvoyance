# -*- coding: utf-8 -*-
"""
[20180320 Treatment Effects with RNNs] sim_seq2seq_training
Created on 4/5/2018 1:17 PM
Train sequence to sequence model
@author: limsi
"""

import treatments.RMSN.configs
import tensorflow as tf
import numpy as np
import logging
import os

from treatments.RMSN.core_routines import train, test
import treatments.RMSN.core_routines as core

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

# EDIT ME! ######################################################################################
# Defines specific parameters to train for - otherwise runs full hyperparameter optimisation

# Specify which networks to train - only use R-MSN by default. Full list in activation map
net_names = ["rnn_propensity_weighted"]


##################################################################################################


# Data processing Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def process_seq_data(
    data_map, states, projection_horizon, num_features_to_include=1e6
):  # forecast 10 years into the future
    def _check_shapes(array1, array2, dims):
        a1_shape = array1.shape
        a2_shape = array2.shape

        if len(a1_shape) != len(a2_shape):
            return False

        b_same = True
        for i in range(dims):
            b_same = b_same and a1_shape[i] == a2_shape[i]

        return b_same

    outputs = data_map["scaled_outputs"]
    sequence_lengths = data_map["sequence_lengths"]
    active_entries = data_map["active_entries"]
    actions = data_map["actions"]
    inputs = data_map["scaled_inputs"][:, :, :num_features_to_include]

    # Check that states are indeed valid
    if not _check_shapes(outputs, states, 2):
        raise ValueError("States and outputs have different shapes!!")

    num_patients, num_time_steps, num_features = outputs.shape

    num_seq2seq_rows = num_patients * num_time_steps

    seq2seq_state_inits = np.zeros((num_seq2seq_rows, states.shape[-1]))
    seq2seq_actions = np.zeros((num_seq2seq_rows, projection_horizon, actions.shape[-1]))
    seq2seq_inputs = np.zeros((num_seq2seq_rows, projection_horizon, inputs.shape[-1]))
    seq2seq_outputs = np.zeros(
        (num_seq2seq_rows, projection_horizon, outputs.shape[-1])
    )  # we reconstruct raw outputs later
    seq2seq_active_entries = np.zeros((num_seq2seq_rows, projection_horizon, active_entries.shape[-1]))
    seq2seq_sequence_lengths = np.zeros(num_seq2seq_rows)

    total_seq2seq_rows = 0  # we use this to shorten any trajectories later

    for i in range(num_patients):
        sequence_length = int(sequence_lengths[i])

        for t in range(1, sequence_length):  # shift outputs back by 1
            seq2seq_state_inits[total_seq2seq_rows, :] = states[i, t - 1, :]  # previous state output

            max_projection = min(projection_horizon, sequence_length - t)
            seq2seq_active_entries[total_seq2seq_rows, :max_projection, :] = active_entries[
                i, t : t + max_projection, :
            ]
            seq2seq_actions[total_seq2seq_rows, :max_projection, :] = actions[i, t : t + max_projection, :]
            seq2seq_outputs[total_seq2seq_rows, :max_projection, :] = outputs[i, t : t + max_projection, :]
            seq2seq_sequence_lengths[total_seq2seq_rows] = max_projection
            seq2seq_inputs[total_seq2seq_rows, :max_projection, :] = inputs[i, t : t + max_projection, :]

            total_seq2seq_rows += 1

    # Filter everything shorter
    seq2seq_state_inits = seq2seq_state_inits[:total_seq2seq_rows, :]
    seq2seq_actions = seq2seq_actions[:total_seq2seq_rows, :, :]
    seq2seq_outputs = seq2seq_outputs[:total_seq2seq_rows, :, :]
    seq2seq_active_entries = seq2seq_active_entries[:total_seq2seq_rows, :, :]
    seq2seq_sequence_lengths = seq2seq_sequence_lengths[:total_seq2seq_rows]
    seq2seq_inputs = seq2seq_inputs[:total_seq2seq_rows, :, :]

    # Package outputs
    seq2seq_data_map = {
        "initial_states": seq2seq_state_inits,
        "scaled_inputs": seq2seq_actions,
        "scaled_outputs": seq2seq_outputs,
        "sequence_lengths": seq2seq_sequence_lengths,
        "active_entries": seq2seq_active_entries,
    }

    # Add propensity weights if they exist
    if "propensity_weights" in data_map:
        count_idx = 0
        propensity_weights = data_map["propensity_weights"]
        seq2seq_propensity_weights = np.zeros((total_seq2seq_rows, projection_horizon, propensity_weights.shape[-1]))
        for i in range(num_patients):

            sequence_length = int(sequence_lengths[i])

            for t in range(1, sequence_length):  # shift outputs back by 1
                max_projection = min(projection_horizon, sequence_length - t)
                ws = propensity_weights[i, t - 1 : t + max_projection, :].cumprod(axis=0)
                seq2seq_propensity_weights[count_idx, :max_projection, :] = ws[1 : max_projection + 1, :]

                count_idx += 1

        # Normalise these weights
        prop_weight_means = np.sum(seq2seq_propensity_weights * seq2seq_active_entries, axis=0) / np.sum(
            seq2seq_active_entries, axis=0
        )

        seq2seq_propensity_weights = seq2seq_propensity_weights / prop_weight_means

        seq2seq_data_map["propensity_weights"] = seq2seq_propensity_weights

    return seq2seq_data_map


# In[*]: Main routine
def decoder_fit(hyperparams_decoder_iptw, training_dataset, validation_dataset, projection_horizon, task, MODEL_ROOT):
    expt_name = "treatment_effects"
    logging.info("Running hyperparameter optimisation")

    # Optimal encoder to load for decoder training
    # - This allows for states from the encoder to be dumped, and decoder is intialised with them
    encoder_specifications = {
        "rnn_propensity_weighted": treatments.RMSN.configs.load_optimal_parameters(
            "rnn_propensity_weighted", expt_name, MODEL_ROOT
        )
    }

    decoder_specifications = {
        "rnn_propensity_weighted": (
            hyperparams_decoder_iptw["dropout_rate"],
            hyperparams_decoder_iptw["memory_multiplier"],
            hyperparams_decoder_iptw["num_epochs"],
            hyperparams_decoder_iptw["batch_size"],
            hyperparams_decoder_iptw["learning_rate"],
            hyperparams_decoder_iptw["max_norm"],
        ),
    }

    # Setup params for datas
    tf_device = "gpu"
    b_apply_memory_adapter = True
    b_single_layer = True  # single layer for memory adapter
    specified_hyperparam_iterations = 20

    if task == "classification":
        activation_map = {
            "rnn_propensity_weighted": ("tanh", "sigmoid"),
            # 'rnn_propensity_weighted_spec': ("elu", 'linear'),
            # 'rnn_propensity_weighted_den_only': ("elu", 'linear'),
            # 'rnn_propensity_weighted_logistic': ("elu", 'linear'),
            "rnn_model": ("tanh", "sigmoid"),
            "treatment_rnn": ("tanh", "sigmoid"),
            "treatment_rnn_action_inputs_only": ("tanh", "sigmoid"),
        }
    else:
        activation_map = {
            "rnn_propensity_weighted": ("elu", "linear"),
            # 'rnn_propensity_weighted_spec': ("elu", 'linear'),
            # 'rnn_propensity_weighted_den_only': ("elu", 'linear'),
            # 'rnn_propensity_weighted_logistic': ("elu", 'linear'),
            "rnn_model": ("elu", "linear"),
            "treatment_rnn": ("tanh", "sigmoid"),
            "treatment_rnn_action_inputs_only": ("tanh", "sigmoid"),
        }

    # Setup tensorflow
    if tf_device == "cpu":
        tf_config = tf.ConfigProto(log_device_placement=False, device_count={"GPU": 0})
    else:
        tf_config = tf.ConfigProto(log_device_placement=False, device_count={"GPU": 1})
        tf_config.gpu_options.allow_growth = True

    # Load biased observational data for training
    b_load = False
    b_save = True

    training_data = training_dataset
    validation_data = validation_dataset
    test_data = training_dataset

    # Start Running hyperparam opt
    opt_params = {}
    for net_name in net_names:
        max_hyperparam_runs = specified_hyperparam_iterations if net_name not in decoder_specifications else 1

        # In[*]: Prep data

        # Pull datasets
        b_predict_actions = "treatment_rnn" in net_name
        use_truncated_bptt = net_name != "rnn_model_bptt"
        b_propensity_weight = "rnn_propensity_weighted" in net_name
        b_use_actions_only = "rnn_action_inputs_only" in net_name

        # Extract only relevant trajs and shift data
        training_processed = core.get_processed_data(training_data, b_predict_actions, b_use_actions_only)
        validation_processed = core.get_processed_data(validation_data, b_predict_actions, b_use_actions_only)
        test_processed = core.get_processed_data(test_data, b_predict_actions, b_use_actions_only)

        num_features = training_processed["scaled_inputs"].shape[-1]  # 4 if not b_use_actions_only else 3
        num_outputs = training_processed["scaled_outputs"].shape[-1]  # 1 if not b_predict_actions else 3    # 5

        # Load propensity weights if they exist
        if b_propensity_weight:
            # raise NotImplementedError("Propensity weights will be added later")

            if net_name == "rnn_propensity_weighted_den_only":
                propensity_weights = np.load(os.path.join(MODEL_ROOT, "propensity_scores_den_only.npy"))
            elif net_name == "rnn_propensity_weighted_logistic":
                propensity_weights = np.load(os.path.join(MODEL_ROOT, "propensity_scores.npy"))
                tmp = np.load(os.path.join(MODEL_ROOT, "propensity_scores_logistic.npy"))
                propensity_weights = tmp[: propensity_weights.shape[0], :, :]
            else:
                propensity_weights = np.load(os.path.join(MODEL_ROOT, "propensity_scores.npy"))

            training_processed["propensity_weights"] = propensity_weights

        # In[*]: Get initial states & transform data maps

        logging.info("Loading basic network to generate states: {}".format(net_name))

        if net_name not in encoder_specifications:
            raise ValueError("Can't find term in hyperparameter specifications")

        spec = encoder_specifications[net_name]
        logging.info("Using specifications for {}: {}".format(net_name, spec))
        dropout_rate = spec[0]
        memory_multiplier = spec[1] / num_features
        num_epochs = spec[2]
        minibatch_size = spec[3]
        learning_rate = spec[4]
        max_norm = spec[5]

        hidden_activation, output_activation = activation_map[net_name]

        model_folder = os.path.join(MODEL_ROOT, net_name)
        train_preds, _, _, train_states = test(
            training_processed,
            tf_config,
            net_name,
            expt_name,
            dropout_rate,
            num_features,
            num_outputs,
            memory_multiplier,
            num_epochs,
            minibatch_size,
            learning_rate,
            max_norm,
            hidden_activation,
            output_activation,
            model_folder,
            b_use_state_initialisation=False,
            b_dump_all_states=True,
        )

        valid_preds, _, _, valid_states = test(
            validation_processed,
            tf_config,
            net_name,
            expt_name,
            dropout_rate,
            num_features,
            num_outputs,
            memory_multiplier,
            num_epochs,
            minibatch_size,
            learning_rate,
            max_norm,
            hidden_activation,
            output_activation,
            model_folder,
            b_use_state_initialisation=False,
            b_dump_all_states=True,
        )

        test_preds, _, _, test_states = test(
            test_processed,
            tf_config,
            net_name,
            expt_name,
            dropout_rate,
            num_features,
            num_outputs,
            memory_multiplier,
            num_epochs,
            minibatch_size,
            learning_rate,
            max_norm,
            hidden_activation,
            output_activation,
            model_folder,
            b_use_state_initialisation=False,
            b_dump_all_states=True,
        )

        # Repackage inputs here
        training_processed = process_seq_data(
            training_processed, train_states, projection_horizon=projection_horizon, num_features_to_include=num_outputs
        )
        validation_processed = process_seq_data(
            validation_processed,
            valid_states,
            projection_horizon=projection_horizon,
            num_features_to_include=num_outputs,
        )
        test_processed = process_seq_data(
            test_processed, test_states, projection_horizon=projection_horizon, num_features_to_include=num_outputs
        )

        encoder_state_size = num_features * memory_multiplier * 2

        num_features = training_processed["scaled_inputs"].shape[-1]
        num_outputs = training_processed["scaled_outputs"].shape[-1]

        # In[*]:

        hyperparam_count = 0
        while True:

            if net_name not in decoder_specifications:

                dropout_rate = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
                memory_multiplier = np.random.choice([1, 2, 4, 8, 16]) if b_apply_memory_adapter else 1
                adapter_multiplier = (
                    np.random.choice([0.25, 0.5, 1, 1.5, 2.0]) if not b_single_layer and b_apply_memory_adapter else 0
                )
                num_epochs = 100
                minibatch_size = np.random.choice([256, 512, 1024])
                learning_rate = np.random.choice([0.01, 0.005, 0.001, 1e-4])  # ([0.01, 0.001, 0.0001])
                max_norm = np.random.choice([0.5, 1.0, 2.0, 4.0])
                hidden_activation, output_activation = activation_map[net_name]

            else:

                spec = decoder_specifications[net_name]
                logging.info("Using specifications for {}: {}".format(net_name, spec))
                dropout_rate = spec[0]
                memory_multiplier = spec[1] / num_features
                adapter_multiplier = 0 if len(spec) < 7 else spec[7] / encoder_state_size
                num_epochs = spec[2]
                minibatch_size = spec[3]
                learning_rate = spec[4]
                max_norm = spec[5]
                hidden_activation, output_activation = activation_map[net_name]

            suffix = "_seq2seq"
            suffix += "_no_adapter" if not b_apply_memory_adapter else ""
            suffix += "_multi_layer" if not b_single_layer else ""
            seq_net_name = net_name + suffix
            model_folder = os.path.join(MODEL_ROOT, seq_net_name)
            hyperparam_opt = train(
                seq_net_name,
                expt_name,
                training_processed,
                validation_processed,
                dropout_rate,
                memory_multiplier,
                num_epochs,
                minibatch_size,
                learning_rate,
                max_norm,
                use_truncated_bptt,
                num_features,
                num_outputs,
                model_folder,
                hidden_activation,
                output_activation,
                tf_config,
                "hyperparam opt: {} of {}".format(hyperparam_count, max_hyperparam_runs),
                b_use_state_initialisation=True,
                b_use_seq2seq_training_mode=False,  # don't loop back outputs into inputs
                adapter_multiplier=adapter_multiplier,
                b_use_memory_adapter=b_apply_memory_adapter,
            )

            hyperparam_count = len(hyperparam_opt.columns)

            if hyperparam_count == max_hyperparam_runs:
                opt_params[seq_net_name] = hyperparam_opt.T
                break

        logging.info("Done")
        logging.info(hyperparam_opt.T)

        # Flag optimal params
