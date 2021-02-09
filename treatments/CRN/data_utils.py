import numpy as np

from utils.data_utils import concate_xs, concate_xt


def data_preprocess_counterfactuals(
    encoder_model, dataset, patient_id, timestep, treatment_options, fold, split, static_mode, time_mode
):
    """Preprocess the dataset for obtaining counterfactual predictions for sequences of future treatments.

    Args:
        - encoder_model: trained encoder model for initializing decoder
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
        - dataset_crn_decoder: dataset that can be used to obtain the counterfactual predictions from the decoder model.
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

    dataset_crn_encoder = dict()

    one_hot_treatments = np.zeros(shape=(treat.shape[0], treat.shape[1], 2))
    treat = np.round(treat)
    for patient_id in range(treat.shape[0]):
        for t in range(treat.shape[1]):
            if treat[patient_id][t][0] == 0.0:
                one_hot_treatments[patient_id][t] = [1, 0]
            elif treat[patient_id][t][0] == 1.0:
                one_hot_treatments[patient_id][t] = [0, 1]
            elif treat[patient_id][t][0] == -1.0:
                one_hot_treatments[patient_id][t] = [-1, -1]

    one_hot_treatments_encoder = one_hot_treatments[:, :timestep, :]
    one_hot_treatments_encoder = np.concatenate(
        [
            one_hot_treatments_encoder,
            np.zeros(shape=(one_hot_treatments.shape[0], max_sequence_length - timestep, one_hot_treatments.shape[-1])),
        ],
        axis=1,
    )

    dataset_crn_encoder["current_covariates"] = x
    dataset_crn_encoder["current_treatments"] = one_hot_treatments_encoder
    dataset_crn_encoder["previous_treatments"] = one_hot_treatments_encoder[:, :-1, :]
    dataset_crn_encoder["active_entries"] = np.ones(shape=(x.shape[0], x.shape[1], 1))
    dataset_crn_encoder["sequence_lengths"] = timestep * np.ones(shape=(num_treatment_options))

    test_br_states = encoder_model.get_balancing_reps(dataset_crn_encoder)
    test_encoder_predictions = encoder_model.get_predictions(dataset_crn_encoder)

    dataset_crn_decoder = dict()
    dataset_crn_decoder["init_states"] = test_br_states[:, timestep - 1, :]
    dataset_crn_decoder["encoder_output"] = test_encoder_predictions[:, timestep - 1, :]
    dataset_crn_decoder["current_treatments"] = one_hot_treatments[:, timestep : timestep + projection_horizon, :]
    dataset_crn_decoder["previous_treatments"] = one_hot_treatments[
        :, timestep - 1 : timestep + projection_horizon - 1, :
    ]
    dataset_crn_decoder["active_entries"] = np.ones(shape=(one_hot_treatments.shape[0], one_hot_treatments.shape[1], 1))
    dataset_crn_decoder["sequence_lengths"] = timestep * np.ones(shape=(projection_horizon))

    patient_history = y[0][:timestep]
    encoder_output = test_encoder_predictions[:, timestep - 1 : timestep, :]

    return patient_history, encoder_output, dataset_crn_decoder


def data_preprocess(dataset, fold, split, static_mode, time_mode):
    """Preprocess the dataset.

    Args:
        - dataset: temporal, static, label, time, treatment information
        - fold: Cross validation fold
        - split: 'train', 'valid' or 'test'
        - static_mode: 'concatenate' or None
        - time_mode: 'concatenate' or None

    Returns:
        - dataset_crn: dataset dictionary for training the CRN.
    """
    x, s, y, t, treat = dataset.get_fold(fold, split)

    if static_mode == "concatenate":
        x = concate_xs(x, s)

    if time_mode == "concatenate":
        x = concate_xt(x, t)

    dataset_crn = dict()

    one_hot_treatments = np.zeros(shape=(treat.shape[0], treat.shape[1], 2))
    treat = np.round(treat)
    for patient_id in range(treat.shape[0]):
        for timestep in range(treat.shape[1]):
            if treat[patient_id][timestep][0] == 0.0:
                one_hot_treatments[patient_id][timestep] = [1, 0]
            elif treat[patient_id][timestep][0] == 1.0:
                one_hot_treatments[patient_id][timestep] = [0, 1]
            elif treat[patient_id][timestep][0] == -1.0:
                one_hot_treatments[patient_id][timestep] = [-1, -1]

    active_entries = np.ndarray.max((y >= 0).astype(float), axis=-1)
    sequence_lengths = np.sum(active_entries, axis=1).astype(int)
    active_entries = active_entries[:, :, np.newaxis]

    dataset_crn["current_covariates"] = x
    dataset_crn["current_treatments"] = one_hot_treatments
    dataset_crn["previous_treatments"] = one_hot_treatments[:, :-1, :]
    dataset_crn["outputs"] = y
    dataset_crn["active_entries"] = active_entries
    dataset_crn["sequence_lengths"] = sequence_lengths

    return dataset_crn


def process_seq_data(dataset, states, projection_horizon):
    """Split the sequences in the training data to train the decoder.

    Args:
        - dataset: dataset with training data sequences
        - states: encoder states used to initialize the decoder
        - projection_horizon: number of future timesteps for training decoder

    Returns:
        - dataset for training decoder
    """

    outputs = dataset["outputs"]
    sequence_lengths = dataset["sequence_lengths"]
    active_entries = dataset["active_entries"]
    current_treatments = dataset["current_treatments"]
    previous_treatments = dataset["previous_treatments"]

    num_patients, num_time_steps, num_features = outputs.shape

    num_seq2seq_rows = num_patients * num_time_steps

    seq2seq_state_inits = np.zeros((num_seq2seq_rows, states.shape[-1]))
    seq2seq_previous_treatments = np.zeros((num_seq2seq_rows, projection_horizon, previous_treatments.shape[-1]))
    seq2seq_current_treatments = np.zeros((num_seq2seq_rows, projection_horizon, current_treatments.shape[-1]))
    seq2seq_current_covariates = np.zeros((num_seq2seq_rows, projection_horizon, outputs.shape[-1]))
    seq2seq_outputs = np.zeros((num_seq2seq_rows, projection_horizon, outputs.shape[-1]))
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
            seq2seq_previous_treatments[total_seq2seq_rows, :max_projection, :] = previous_treatments[
                i, t - 1 : t + max_projection - 1, :
            ]
            seq2seq_current_treatments[total_seq2seq_rows, :max_projection, :] = current_treatments[
                i, t : t + max_projection, :
            ]
            seq2seq_outputs[total_seq2seq_rows, :max_projection, :] = outputs[i, t : t + max_projection, :]
            seq2seq_sequence_lengths[total_seq2seq_rows] = max_projection
            seq2seq_current_covariates[total_seq2seq_rows, :max_projection, :] = outputs[
                i, t - 1 : t + max_projection - 1, :
            ]

            total_seq2seq_rows += 1

    # Filter everything shorter
    seq2seq_state_inits = seq2seq_state_inits[:total_seq2seq_rows, :]
    seq2seq_previous_treatments = seq2seq_previous_treatments[:total_seq2seq_rows, :, :]
    seq2seq_current_treatments = seq2seq_current_treatments[:total_seq2seq_rows, :, :]
    seq2seq_current_covariates = seq2seq_current_covariates[:total_seq2seq_rows, :, :]
    seq2seq_outputs = seq2seq_outputs[:total_seq2seq_rows, :, :]
    seq2seq_active_entries = seq2seq_active_entries[:total_seq2seq_rows, :, :]
    seq2seq_sequence_lengths = seq2seq_sequence_lengths[:total_seq2seq_rows]

    # Package outputs
    seq2seq_data_map = {
        "init_state": seq2seq_state_inits,
        "previous_treatments": seq2seq_previous_treatments,
        "current_treatments": seq2seq_current_treatments,
        "current_covariates": seq2seq_current_covariates,
        "outputs": seq2seq_outputs,
        "sequence_lengths": seq2seq_sequence_lengths,
        "active_entries": seq2seq_active_entries,
    }

    return seq2seq_data_map
