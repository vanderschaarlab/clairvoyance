from __future__ import absolute_import, division, print_function

import streamlit as st
import pandas as pd
import numpy as np
import copy as cp
import io as io

APP_VERSION = 0.1

AUTO_UPLOAD = True  # NOTE: Deploy @ False

#########################################################################################
# Imports
#########################################################################################

import warnings

warnings.filterwarnings("ignore")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (1) Upload Dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from datasets import CSVLoader

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (2) Preprocess Dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from preprocessing import FilterNegative
from preprocessing import OneHotEncoder
from preprocessing import MinMaxNormalizer
from preprocessing import StandardNormalizer

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (3) Define Problem
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from preprocessing import ProblemMaker

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (4) Impute Dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from imputation import BasicImputation
from imputation import Interpolation
from imputation import NNImputation
from imputation import StandardImputation

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (5) Fit and Predict
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from prediction import Attention
from prediction import GeneralRNN
from prediction import TemporalCNN
from prediction import TransformerPredictor

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (6) Estimate Uncertainty
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from uncertainty import EnsembleUncertainty

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (7) Interpret Predictions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from interpretation import TInvase

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (8) Visualize Results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from evaluation import Metrics
from evaluation import print_performance

# from evaluation     import print_prediction
from evaluation import print_uncertainty
from evaluation import print_interpretation

from utils import compose

#########################################################################################
# View
#########################################################################################

st.title("MediTime")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Badges
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

st.markdown(
    """
<img alt="Awesome" src="https://badgen.net/badge/icon/awesome/CC0066?icon=awesome&label">
<img alt="Awesome" src="https://badgen.net/badge/icon/github/E68718?icon=github&label">
<img alt="Awesome" src="https://badgen.net/badge/release/v
"""
    + str(APP_VERSION)
    + """
/428F7E">
<img alt="Awesome" src="https://badgen.net/badge/license/MIT/blue">
""",
    unsafe_allow_html=True,
)

page_zoom = """
<style>
    html {
        zoom: 95%;
    }
</style>
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Banner
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ban_style = """
<style>
    header .decoration {
        height: 16px;
        opacity: 0.25;
        background-image: linear-gradient(90deg,#ff0066,#f0f2f6,#0000ff);

        text-align: center;
        font-size: 0.5em;
        color: black;
    }

    header .decoration:empty:before {
        content: '';
    }

    header .toolbar {
        top: 1rem;
    }

    .reportview-container {
        top: 1rem;
    }

    .reportview-container .sidebar .sidebar-collapse-control {
        top: 1.5rem;
    }

    .reportview-container .sidebar .sidebar-content .block-container {
        margin-left:  20px;
        margin-right: 20px;
    }
</style>
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Footer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

bot_style = """
<style>
    .reportview-container .main footer {
        visibility: hidden;
    }
</style>
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Burger
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nav_style = """
<style>
    #MainMenu .dropdown-menu-right button:nth-of-type(4) {
        display: none;
    }

    #MainMenu .dropdown-menu-right button:nth-of-type(5) {
        display: none;
    }

    #MainMenu .dropdown-menu-right button:nth-of-type(6) {
        display: none;
    }

    #MainMenu .dropdown-menu-right button:nth-of-type(7) {
        display: none;
    }

    #MainMenu .dropdown-menu-right button:nth-of-type(9) {
        display: none;
    }
</style>
"""

div_style = """
<style>
    #MainMenu .dropdown-menu-right div:nth-of-type(1) {
        display: none;
    }

    #MainMenu .dropdown-menu-right div:nth-of-type(2) {
        display: none;
    }
</style>
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Notes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

done_text = "Commit"

done_note = "Check to commit your selections."

data_note = "Choose a file (.csv, max 100 MB):"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Consts
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

display_lim = 10_000

hash_funcs = {
    io.StringIO: io.StringIO.getvalue,
}

types = [
    "csv",
    "gz",
]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Write
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

st.markdown(page_zoom + ban_style + bot_style + nav_style + div_style, unsafe_allow_html=True)

#########################################################################################
# (1) Upload Dataset
#########################################################################################

st.header("Step 1. Upload Dataset")
st.sidebar.subheader("Step 1. Upload Dataset")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Handler
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@st.cache(hash_funcs=hash_funcs,)
def upload_dataset(csv):
    data = pd.read_csv(csv, compression="infer",)

    return data


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Body (A)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

st.subheader("Static Data")

st.text("Training:")

if not AUTO_UPLOAD:
    csv_stat_t = st.file_uploader(label=data_note, type=types, key="csv_stat_t",)
else:
    csv_stat_t = "datasets/data/mimic/repl/csv/mimic_static_train_data.csv"

if csv_stat_t:
    ind_stat_b = st.sidebar.text("Uploading train (stat)...")
    dat_stat_t = upload_dataset(csv_stat_t)
    ind_stat_b.text("Uploading train (stat)... OK.")

    st.write(dat_stat_t[:display_lim])

st.text("Testing:")

if not AUTO_UPLOAD:
    csv_stat_v = st.file_uploader(label=data_note, type=types, key="csv_stat_v",)
else:
    csv_stat_v = "datasets/data/mimic/repl/csv/mimic_static_test_data.csv"

if csv_stat_v:
    ind_stat_b.text("Uploading test (stat)...")
    dat_stat_v = upload_dataset(csv_stat_v)
    ind_stat_b.text("Uploading test (stat)... OK.")

    st.write(dat_stat_v[:display_lim])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Body (B)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

st.subheader("Temporal Data")

st.text("Training:")

if not AUTO_UPLOAD:
    csv_temp_t = st.file_uploader(label=data_note, type=types, key="csv_temp_t",)
else:
    csv_temp_t = "datasets/data/mimic/repl/csv/mimic_temporal_train_data_eav.csv"

if csv_temp_t:
    ind_temp_b = st.sidebar.text("Uploading train (temp)...")
    dat_temp_t = upload_dataset(csv_temp_t)
    ind_temp_b.text("Uploading train (temp)... OK.")

    st.write(dat_temp_t[:display_lim])

st.text("Testing:")

if not AUTO_UPLOAD:
    csv_temp_v = st.file_uploader(label=data_note, type=types, key="csv_temp_v",)
else:
    csv_temp_v = "datasets/data/mimic/repl/csv/mimic_temporal_test_data_eav.csv"

if csv_temp_v:
    ind_temp_b.text("Uploading test (temp)...")
    dat_temp_v = upload_dataset(csv_temp_v)
    ind_temp_b.text("Uploading test (temp)... OK.")

    st.write(dat_temp_v[:display_lim])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Wrapper
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@st.cache(allow_output_mutation=True,)
def load_dataset():
    data_loader_t = CSVLoader(static_file=csv_stat_t, temporal_file=csv_temp_t,)

    data_loader_v = CSVLoader(static_file=csv_stat_v, temporal_file=csv_temp_v,)

    return (
        data_loader_t.load(),
        data_loader_v.load(),
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if csv_stat_t and csv_stat_v and csv_temp_t and csv_temp_v:
    st.markdown("<hr>", unsafe_allow_html=True)
    step_1 = st.checkbox(done_text, key="step_1")

    if step_1:
        step_1_text = st.sidebar.text("Preparing data loader...")
        dataset_t_1, dataset_v_1 = load_dataset()
        step_1_text.text("Preparing data loader... done.")
    else:
        st.info(done_note)
else:
    step_1 = False

#########################################################################################
# (2) Preprocess Dataset
#########################################################################################

if step_1:
    st.header("Step 2. Preprocess Dataset")
    st.sidebar.subheader("Step 2. Preprocess Dataset")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Names
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    cols_stat = [col for col in dat_stat_v if not col == "id"]
    cols_temp = [var for var in dat_temp_v.variable.unique()]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Body (A)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    st.subheader("Negative Values")

    negative_filter = st.checkbox("Replace negative values with NaN", key="filter_negative")

    st.sidebar.text(("F" if negative_filter else "Not f") + "iltering negative values.")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Body (B)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    st.subheader("Data Normalization")

    norm_dict = {
        "No normalization": None,
        "Min-max normalization": MinMaxNormalizer(),
        "Standard normalization": StandardNormalizer(),
    }

    select_norm = st.radio("Choose a data normalization method:", list(norm_dict.keys()), index=0, key="step_2_radio")

    st.sidebar.text(select_norm + " selected.")

    feature_normalizer = norm_dict[select_norm]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Body (C)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    st.subheader("One-Hot Encoding")

    ohot_stat = st.multiselect("Choose one or more static variables to one-hot encode:", cols_stat)

    if ohot_stat:
        st.sidebar.text("Encoding (stat): " + (", ").join(ohot_stat) + ".")

    ohot_temp = st.multiselect("Choose one or more temporal variables to one-hot encode:", cols_temp)

    if ohot_temp:
        st.sidebar.text("Encoding (temp): " + (", ").join(ohot_temp) + ".")

    onehot_encoder = OneHotEncoder(one_hot_encoding_features=ohot_stat + ohot_temp)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Wrapper
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@st.cache
def preprocess_dataset(
    dataset_t, dataset_v,
):
    if negative_filter:
        filter_pipeline = compose(FilterNegative().fit_transform, onehot_encoder.fit_transform,)
    else:
        filter_pipeline = compose(onehot_encoder.fit_transform,)

    return (
        (
            feature_normalizer.fit_transform(filter_pipeline(dataset_t)),
            feature_normalizer.fit_transform(filter_pipeline(dataset_v)),
        )
        if feature_normalizer
        else (filter_pipeline(dataset_t), filter_pipeline(dataset_v),)
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if step_1:
    st.markdown("<hr>", unsafe_allow_html=True)
    step_2 = st.checkbox(done_text, key="step_2")

    if step_2:
        step_2_text = st.sidebar.text("Preprocessing data...")

        dataset_t_1_copy = cp.deepcopy(dataset_t_1)
        dataset_v_1_copy = cp.deepcopy(dataset_v_1)
        dataset_t_2, dataset_v_2 = preprocess_dataset(dataset_t_1_copy, dataset_v_1_copy,)

        step_2_text.text("Preprocessing data... done.")
    else:
        st.info(done_note)

#########################################################################################
# (3) Define Problem
#########################################################################################

if step_1 and step_2:
    st.header("Step 3. Define Problem")
    st.sidebar.subheader("Step 3. Define Problem")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Names
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    cols_stat = [col for col in dataset_t_2.static_data if not col == "id"]
    cols_temp = [col for col in dataset_t_2.temporal_data if not col in ["id", "time"]]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Body (A)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    st.subheader("Problem Type")

    problem_dict = {
        "One-shot problem (stat endpoint)": "one-shot",
        "Online problem (temp endpoint)": "online",
    }

    select_type = st.radio("Choose a problem type:", list(problem_dict.keys()), index=0, key="step_3_radio")

    st.sidebar.text(select_type + ".")

    problem_type = problem_dict[select_type]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Body (B)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    st.subheader("Prediction Endpoint")

    if problem_type == "one-shot":
        output_var = st.selectbox("Choose static label (i.e. output variable):", cols_stat)

        if output_var:
            st.sidebar.text("Predict (stat): " + output_var + ".")

        label_name = [output_var]

    if problem_type == "online":
        output_var = st.selectbox("Choose temporal label (i.e. output variable):", cols_temp)

        if output_var:
            st.sidebar.text("Predict (temp): " + output_var + ".")

        label_name = [output_var]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Body (C)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    st.subheader("Sequence Length")

    max_seq_len = st.slider(
        label="Select maximum sequence length:", min_value=1, max_value=24, value=24, step=1, format="%d",
    )

    st.sidebar.text("Maximum sequence length: " + str(max_seq_len) + ".")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Wrapper
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@st.cache
def define_problem(
    dataset_t, dataset_v,
):
    problem_maker = ProblemMaker(problem=problem_type, label=label_name, max_seq_len=max_seq_len, treatment=None,)

    return (
        problem_maker.fit_transform(dataset_t),
        problem_maker.fit_transform(dataset_v),
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if step_1 and step_2:
    st.markdown("<hr>", unsafe_allow_html=True)
    step_3 = st.checkbox(done_text, key="step_3")

    if step_3:
        step_3_text = st.sidebar.text("Defining problem...")

        dataset_t_2_copy = cp.deepcopy(dataset_t_2)
        dataset_v_2_copy = cp.deepcopy(dataset_v_2)
        dataset_t_3, dataset_v_3 = define_problem(dataset_t_2_copy, dataset_v_2_copy,)

        step_3_text.text("Defining problem... done.")
    else:
        st.info(done_note)

#########################################################################################
# (4) Impute Dataset
#########################################################################################

if step_1 and step_2 and step_3:
    st.header("Step 4. Impute Dataset")
    st.sidebar.subheader("Step 4. Impute Dataset")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Body (A)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    st.subheader("Static Imputation")

    stat_impute_dict = {
        "Mean imputation": "mean",
        "Median imputation": "median",
        "MICE imputation": "mice",
        "MissForest imputation": "missforest",
        "KNN imputation": "knn",
        "GAIN imputation": "gain",
    }

    select_stat_impute = st.radio(
        "Choose a data imputation method:", list(stat_impute_dict.keys()), index=0, key="step_4_radio_stat"
    )

    st.sidebar.text(select_stat_impute + " (stat).")

    stat_impute_type = stat_impute_dict[select_stat_impute]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Body (B)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    st.subheader("Temporal Imputation")

    temp_impute_dict = {
        "Mean imputation": "mean",
        "Median imputation": "median",
        "Linear interpolation": "linear",
        "Quadratic interpolation": "quadratic",
        "Cubic interpolation": "cubic",
        "Spline interpolation": "spline",
        "MRNN imputation": "mrnn",
        "TGAIN imputation": "tgain",
    }

    select_temp_impute = st.radio(
        "Choose a data imputation method:", list(temp_impute_dict.keys()), index=0, key="step_4_radio_temp"
    )

    st.sidebar.text(select_temp_impute + " (temp).")

    temp_impute_type = temp_impute_dict[select_temp_impute]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Wrapper
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@st.cache
def impute_dataset(
    dataset_t, dataset_v,
):
    if stat_impute_type in ["mean", "median"]:
        stat_impute_func = BasicImputation

    if stat_impute_type in ["mice", "missforest", "knn"]:
        stat_impute_func = StandardImputation

    if stat_impute_type in ["gain"]:
        stat_impute_func = NNImputation

    if temp_impute_type in ["mean", "median"]:
        temp_impute_func = BasicImputation
        temp_impute_dict = {
            "imputation_model_name": temp_impute_type,
            "data_type": "temporal",
        }

    if temp_impute_type in ["linear", "quadratic", "cubic", "spline"]:
        temp_impute_func = Interpolation
        temp_impute_dict = {
            "interpolation_model_name": temp_impute_type,
            "data_type": "temporal",
        }

    if temp_impute_type in ["mrnn", "tgain"]:
        temp_impute_func = NNImputation
        temp_impute_dict = {
            "imputation_model_name": temp_impute_type,
            "data_type": "temporal",
        }

    static_imputation = stat_impute_func(imputation_model_name=stat_impute_type, data_type="static",)

    temporal_imputation = temp_impute_func(**temp_impute_dict)

    imputation_pipeline = compose(static_imputation.fit_transform, temporal_imputation.fit_transform,)

    return (
        imputation_pipeline(dataset_t),
        imputation_pipeline(dataset_v),
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if step_1 and step_2 and step_3:
    st.markdown("<hr>", unsafe_allow_html=True)
    step_4 = st.checkbox(done_text, key="step_4")

    if step_4:
        step_4_text = st.sidebar.text("Imputing dataset...")

        dataset_t_3_copy = cp.deepcopy(dataset_t_3)
        dataset_v_3_copy = cp.deepcopy(dataset_v_3)
        dataset_t_4, dataset_v_4 = impute_dataset(dataset_t_3_copy, dataset_v_3_copy,)

        step_4_text.text("Imputing dataset... done.")
    else:
        st.info(done_note)

#########################################################################################
# (5) Fit and Predict
#########################################################################################

if step_1 and step_2 and step_3 and step_4:
    st.header("Step 5. Fit and Predict")
    st.sidebar.subheader("Step 5. Fit and Predict")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Body (A)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    st.subheader("Prediction Task")

    pred_task_dict = {
        "Classification": "classification",
        "Regression": "regression",
    }

    select_pred_task = st.radio(
        "Choose a prediction task:", list(pred_task_dict.keys()), index=0, key="step_5_radio_task"
    )

    st.sidebar.text("Task selected: " + select_pred_task + ".")

    pred_task = pred_task_dict[select_pred_task]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Body (B)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    st.subheader("Prediction Method")

    pred_meth_dict = {
        "RNN Model": "rnn",
        "GRU Model": "gru",
        "LSTM Model": "lstm",
        "Attention Model": "attention",
        "TCN Model": "tcn",
        "Transformer Model": "transformer",
    }

    select_pred_meth = st.radio(
        "Choose a prediction method: *", list(pred_meth_dict.keys()), index=0, key="step_5_radio_meth"
    )

    st.sidebar.text("Method selected: " + select_pred_meth + ".")

    pred_meth = pred_meth_dict[select_pred_meth]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Body (C)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    st.subheader("Uncertainty Estimation")

    unct_meth_dict = {
        "Ensemble": "ensemble",
    }

    select_unct_meth = st.radio(
        "Choose an uncertainty estimation method: *", list(unct_meth_dict.keys()), index=0, key="step_6_radio_meth"
    )

    st.sidebar.text("Uncertainty estimation: " + select_unct_meth + ".")

    unct_meth = unct_meth_dict[select_unct_meth]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Body (D)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    st.subheader("Interpretation Method")

    intr_meth_dict = {
        "INVASE": TInvase,
    }

    select_intr_meth = st.radio(
        "Choose an interpretation method: *", list(intr_meth_dict.keys()), index=0, key="step_7_radio_meth"
    )

    st.sidebar.text("Interpretation method: " + select_intr_meth + ".")

    intr_meth = intr_meth_dict[select_intr_meth]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parameters
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    st.text("*Sensible hyperparameters will be selected for you.")

    model_parameters = {
        "h_dim": 100,
        "n_layer": 2,
        "batch_size": 400,
        "epoch": 2,  # NOTE: Deploy @ 20
        "model_type": pred_meth,
        "learning_rate": 0.001,
        "static_mode": "concatenate",
        "time_mode": "concatenate",
        "verbose": True,
    }

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Wrapper
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@st.cache(allow_output_mutation=True)
def fit_and_predict(
    dataset_t, dataset_v,
):
    dataset_t.train_val_test_split(
        prob_val=0.2, prob_test=0.0,
    )

    # Step 5: Fit and Predict

    if pred_meth == "attention":
        pred_klass = Attention

    if pred_meth in ["rnn", "lstm", "gru"]:
        pred_klass = GeneralRNN

    if pred_meth == "tcn":
        model_parameters.pop("model_type", None)
        pred_klass = TemporalCNN

    if pred_meth == "transformer":
        model_parameters.pop("model_type", None)
        model_parameters["n_head"] = 2
        pred_klass = TransformerPredictor

    pred_class = pred_klass(task=pred_task)
    pred_class.set_params(**model_parameters)
    pred_class.fit(dataset_t)
    test_y_hat = pred_class.predict(dataset_v)

    # Step 6: Estimate Uncertainty

    if unct_meth == "ensemble":
        uncertainty = EnsembleUncertainty(
            ensemble_model_type=["rnn", "gru"], predictor_model=pred_class, task=pred_task,
        )

    uncertainty.set_params(**model_parameters)
    uncertainty.fit(dataset_t)
    test_c_hat = uncertainty.predict(dataset_v)

    # Step 7: Interpret Predictions

    interpretation = intr_meth(predictor_model=pred_class, task=pred_task,)

    interpretation.set_params(**model_parameters)
    interpretation.fit(dataset_t)
    test_s_hat = interpretation.predict(dataset_v)

    return test_y_hat, test_c_hat, test_s_hat


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if step_1 and step_2 and step_3 and step_4:
    st.markdown("<hr>", unsafe_allow_html=True)
    step_5 = st.checkbox(done_text, key="step_5")

    if step_5:
        step_5_text = st.sidebar.text("Making predictions...")

        dataset_t_4_copy = cp.deepcopy(dataset_t_4)
        dataset_v_4_copy = cp.deepcopy(dataset_v_4)
        dataset_t_5 = dataset_t_4_copy
        dataset_v_5 = dataset_v_4_copy
        test_y_hat, test_c_hat, test_s_hat, = fit_and_predict(dataset_t_5, dataset_v_5,)

        step_5_text.text("Making predictions... done.")
    else:
        st.info(done_note)

#########################################################################################
# (8) Visualize Results
#########################################################################################

if step_1 and step_2 and step_3 and step_4 and step_5:
    st.header("Step 6. Visualize Results")
    st.sidebar.subheader("Step 6. Visualize Results")

    idx = np.random.permutation(len(test_y_hat))[:2]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Body (A)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    st.subheader("Overall Prediction Performance")

    if select_pred_task == "Classification":
        metric_sets = ["auc", "apr"]

    if select_pred_task == "Regression":
        metric_sets = ["mse", "mae"]

    metric_parameters = {
        "problem": problem_type,
        "label_name": label_name,
    }

    metrics = Metrics(metric_sets, metric_parameters)

    result = metrics.evaluate(dataset_v_5.label, test_y_hat)

    if problem_type == "one-shot":
        text = print_performance(result, metric_sets, metric_parameters,)

        st.text(text)

    if problem_type == "online":
        figs = print_performance(result, metric_sets, metric_parameters,)

        for fig in figs:
            st.pyplot(fig, facecolor=fig.get_facecolor(), edgecolor="none")

    st.sidebar.text("Plotting performance... done.")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Body (B)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    st.subheader("Example Predictions and Uncertainties")

    if problem_type == "one-shot":
        text = print_uncertainty(test_y_hat[idx], test_c_hat[idx], metric_parameters,)

        st.text(text)

    if problem_type == "online":
        figs = print_uncertainty(test_y_hat[idx], test_c_hat[idx], metric_parameters,)

        for fig in figs:
            st.pyplot(fig, facecolor=fig.get_facecolor(), edgecolor="none")

    st.sidebar.text("Plotting predictions... done.")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Body (C)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    st.subheader("Example Interpretations")

    figs = print_interpretation(test_s_hat[idx], dataset_t_5.feature_name, metric_parameters, model_parameters,)

    for fig in figs:
        st.pyplot(fig, facecolor=fig.get_facecolor(), edgecolor="none")

    st.sidebar.text("Plotting interpretations... done.")
