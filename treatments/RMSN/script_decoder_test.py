# -*- coding: utf-8 -*-
"""
CODE ADAPTED FROM: https://github.com/sjblim/rmsn_nips_2018

[20180320 Treatment Effects with RNNs] sim_seq2seq_test
Created on 6/5/2018 11:22 AM

@author: limsi
"""

import treatments.RMSN.configs as configs

from treatments.RMSN.configs import load_optimal_parameters

import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import os

from treatments.RMSN.core_routines import test

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

net_names = ["rnn_propensity_weighted"]


##################################################################################################


# In[*]: Main routine
def decoder_test(test_dataset, task, MODEL_ROOT):
    logging.info("Running hyperparameter optimisation")

    expt_name = "treatment_effects"

    # EDIT ME! ######################################################################################
    # Which networks to load for testing
    decoder_specifications = {
        "rnn_propensity_weighted_seq2seq": load_optimal_parameters(
            "rnn_propensity_weighted_seq2seq", expt_name, MODEL_ROOT
        )
    }

    encoder_specifications = {
        "rnn_propensity_weighted": load_optimal_parameters("rnn_propensity_weighted", expt_name, MODEL_ROOT)
    }

    # Setup params for datas
    tf_device = "gpu"
    b_apply_memory_adapter = True
    b_single_layer = True  # single or multilayer memory adapter
    max_coeff = 10

    if task == "classification":
        activation_map = {
            "rnn_propensity_weighted": ("tanh", "sigmoid"),
            # 'rnn_propensity_weighted_den_only': ("elu", 'linear'),
            # 'rnn_propensity_weighted_logistic': ("elu", 'linear'),
            "rnn_model": ("elu", "linear"),
        }
    else:
        activation_map = {
            "rnn_propensity_weighted": ("elu", "linear"),
            # 'rnn_propensity_weighted_den_only': ("elu", 'linear'),
            # 'rnn_propensity_weighted_logistic': ("elu", 'linear'),
            "rnn_model": ("elu", "linear"),
        }

    # Setup tensorflow
    if tf_device == "cpu":
        tf_config = tf.ConfigProto(log_device_placement=False, device_count={"GPU": 0})
    else:
        tf_config = tf.ConfigProto(log_device_placement=False, device_count={"GPU": 1})
        tf_config.gpu_options.allow_growth = True

    # In[*] Start Running testing procedure
    results_map = {}
    projection_map = {}
    for net_name in net_names:

        # Test
        suffix = "_seq2seq"
        suffix += "_no_adapter" if not b_apply_memory_adapter else ""
        suffix += "_multi_layer" if not b_single_layer else ""

        seq_net_name = net_name + suffix
        model_folder = os.path.join(MODEL_ROOT, seq_net_name)

        if seq_net_name not in decoder_specifications:
            raise ValueError("Cannot find decoder specifications for {}".format(seq_net_name))

        results_map[seq_net_name] = pd.DataFrame(
            [], index=[i for i in range(max_coeff + 1)], columns=[i for i in range(max_coeff + 1)]
        )
        projection_map[seq_net_name] = {}

        # Data setup
        test_processed = test_dataset

        num_features = test_processed["scaled_inputs"].shape[-1]
        num_outputs = test_processed["scaled_outputs"].shape[-1]

        # Pulling specs
        spec = decoder_specifications[seq_net_name]
        logging.info("Using specifications for {}: {}".format(seq_net_name, spec))
        dropout_rate = spec[0]
        memory_multiplier = spec[1] / num_features  # hack to recover correct size
        num_epochs = spec[2]
        minibatch_size = spec[3]
        learning_rate = spec[4]
        max_norm = spec[5]
        hidden_activation, output_activation = activation_map[net_name]

        means, _, _, _ = test(
            test_processed,
            tf_config,
            seq_net_name,
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
            b_use_state_initialisation=True,
            b_dump_all_states=False,
            b_mse_by_time=True,
            b_use_memory_adapter=b_apply_memory_adapter,
        )

        test_processed["predicted_outcomes"] = means

        return means
