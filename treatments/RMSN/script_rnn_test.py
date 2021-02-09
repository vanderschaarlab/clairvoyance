"""
CODE ADAPTED FROM: https://github.com/sjblim/rmsn_nips_2018

[20180320 Treatment Effects with RNNs] test_script
"""

import treatments.RMSN.configs as configs

from treatments.RMSN.core_routines import test
import treatments.RMSN.core_routines as core
from treatments.RMSN.configs import load_optimal_parameters

from treatments.RMSN.libs.model_rnn import RnnModel
import treatments.RMSN.libs.net_helpers as helpers

from sklearn.metrics import roc_auc_score, average_precision_score

import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import os

import matplotlib.pyplot as plt
import tensorflow as tf

# import seaborn as sns
# sns.set()

ROOT_FOLDER = configs.ROOT_FOLDER
MODEL_ROOT = configs.MODEL_ROOT
RESULTS_FOLDER = configs.RESULTS_FOLDER

# Default params:


# EDIT ME! ######################################################################################
# Optimal network parameters to load for testing!


##################################################################################################


def rnn_test(dataset_test, task, MODEL_ROOT):
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

    expt_name = "treatment_effects"

    # Setup tensorflow
    tf_device = "gpu"
    if tf_device == "cpu":
        tf_config = tf.ConfigProto(log_device_placement=False, device_count={"GPU": 0})
    else:
        tf_config = tf.ConfigProto(log_device_placement=False, device_count={"GPU": 1})
        tf_config.gpu_options.allow_growth = True

    configs = [load_optimal_parameters("rnn_propensity_weighted", expt_name, MODEL_ROOT, add_net_name=True)]

    # Config
    if task == "classification":
        activation_map = {
            "rnn_propensity_weighted": ("tanh", "sigmoid"),
            "rnn_propensity_weighted_binary": ("tanh", "sigmoid"),
            "rnn_propensity_weighted_logistic": ("tanh", "sigmoid"),
            "rnn_model": ("tanh", "sigmoid"),
            "treatment_rnn": ("tanh", "sigmoid"),
            "treatment_rnn_actions_only": ("tanh", "sigmoid"),
        }
    else:
        activation_map = {
            "rnn_propensity_weighted": ("elu", "linear"),
            "rnn_propensity_weighted_binary": ("elu", "linear"),
            "rnn_propensity_weighted_logistic": ("elu", "linear"),
            "rnn_model": ("elu", "linear"),
            "treatment_rnn": ("tanh", "sigmoid"),
            "treatment_rnn_actions_only": ("tanh", "sigmoid"),
        }

    projection_map = {}
    mse_by_followup = {}
    for config in configs:
        net_name = config[0]
        projection_map[net_name] = {}

        test_data = dataset_test

        # Setup some params
        b_predict_actions = "treatment_rnn" in net_name
        b_propensity_weight = "rnn_propensity_weighted" in net_name
        b_use_actions_only = "treatment_rnn_action_inputs_only" in net_name

        # In[*]: Compute base MSEs
        # Extract only relevant trajs and shift data
        test_processed = core.get_processed_data(test_data, b_predict_actions, b_use_actions_only)

        num_features = test_processed["scaled_inputs"].shape[-1]  # 4 if not b_use_actions_only else 3
        num_outputs = test_processed["scaled_outputs"].shape[-1]  # 1 if not b_predict_actions else 3  # 5

        # Pull remaining params
        dropout_rate = config[1]
        memory_multiplier = config[2] / num_features
        num_epochs = config[3]
        minibatch_size = config[4]
        learning_rate = config[5]
        max_norm = config[6]
        backprop_length = 60  # we've fixed this
        hidden_activation = activation_map[net_name][0]
        output_activation = activation_map[net_name][1]

        # Run tests
        model_folder = os.path.join(MODEL_ROOT, net_name)

        means, output, mse, test_states = test(
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

    return means, test_states
