import numpy as np
import pandas as pd

from utils.data_utils import list_diff

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

from app import app, backend_contents

data_bundle = backend_contents["mimic"]["data"]
problem_bundles = backend_contents["mimic"]["problems"]
patient_ids = list(data_bundle.testing_id_row_map.keys())

layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                html.Span("Patient:", className="clairvoyance-label",),
                                dcc.Dropdown(
                                    id="patient-dropdown",
                                    className="clairvoyance-dropdown",
                                    style={"width": "120px"},
                                    options=[{"label": i, "value": i} for i in patient_ids],
                                    value=patient_ids[0],
                                ),
                            ],
                            className="clairvoyance-top-panel-dropdown",
                        ),
                        html.Div(id="static-data-table"),
                    ]
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                html.Span("Time series:", className="clairvoyance-label",),
                                dcc.Dropdown(
                                    id="ts-data-dropdown",
                                    className="clairvoyance-dropdown",
                                    style={"width": "200px"},
                                    options=[{"label": i, "value": i} for i in data_bundle.temporal_feature_names],
                                    value=data_bundle.temporal_feature_names[0],
                                ),
                            ],
                            className="clairvoyance-top-panel-dropdown",
                        ),
                        html.Div(dcc.Graph(id="ts-data-figure"), style={"margin-top": "5px"}),
                    ]
                ),
            ],
            className="dataset-row",
        ),
        dbc.Tabs(
            [
                dbc.Tab(label="Prediction", tab_id="ts"),
                dbc.Tab(label="Active Sensing", tab_id="as"),
                dbc.Tab(label="Treatment Effects", tab_id="ite"),
            ],
            id="tabs",
            active_tab="ts",
        ),
        dbc.Row(id="tab-content"),
    ]
)

ts_tab = [
    dbc.Col(html.Div(dcc.Graph(id="ts-pred-figure"),)),
    dbc.Col(html.Div(dcc.Graph(id="interpret-figure"),)),
]

as_tab = [
    dbc.Col(html.Div(dcc.Graph(id="as-figure"),)),
]

validation_layout = [layout, ts_tab, as_tab]


@app.callback(
    Output("tab-content", "children"), [Input("tabs", "active_tab")],
)
def render_tab_content(active_tab):
    if active_tab is not None:
        if active_tab == "ts":
            return ts_tab
        elif active_tab == "as":
            return as_tab
    return "No tab selected"


@app.callback(Output("static-data-table", "children"), [Input("patient-dropdown", "value")])
def update_static_data_table(value):
    print("update_static_data_table()...")

    patient_id = value
    df = data_bundle.raw_dataset_testing.static_data
    val = df[df["id"] == patient_id].values.tolist()[0]
    df = pd.DataFrame(data={"Record": data_bundle.raw_dataset_testing.static_data.columns, "Value": val})

    return (
        dbc.Table.from_dataframe(  # pylint: disable=no-member
            df, striped=True, bordered=True, hover=True, className="table-sm"
        ),
    )


@app.callback(
    Output("ts-data-figure", "figure"), [Input("ts-data-dropdown", "value"), Input("patient-dropdown", "value")]
)
def update_ts_data_figure(ts_var, patient_id):
    print("update_ts_data_figure()...")

    df = data_bundle.raw_dataset_testing.temporal_data
    df = df.loc[df["id"] == patient_id, ("time", ts_var)]

    fig = px.scatter(
        x=df["time"],
        y=df[ts_var],
        title=f"Time series: {ts_var}",
        template="plotly_dark",
        labels=dict(x="Hours", y=ts_var),
    )

    fig.update_layout(
        height=400, margin=dict(l=80, r=80, b=10, t=40, pad=0), paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


@app.callback(Output("ts-pred-figure", "figure"), [Input("patient-dropdown", "value")])
def update_ts_pred_figure(patient_id):
    print("update_ts_pred_figure()...")

    problem_bundle = problem_bundles["time_series_prediction"]

    patient_row_idx = data_bundle.testing_id_row_map[patient_id]
    ts_pred = problem_bundle.results.test_y_hat[patient_row_idx].reshape(problem_bundle.problem_maker.max_seq_len)
    ts_error = problem_bundle.results.test_ci_hat[patient_row_idx].reshape(problem_bundle.problem_maker.max_seq_len)
    time = list(range(0, len(ts_pred)))

    fig = px.line(
        x=time,
        y=ts_pred,
        title="Prediction: Ventilator",
        template="plotly_dark",
        labels=dict(x="Hours", y="Ventilator probability"),
        range_x=(min(time), max(time)),
    )

    e_x = time + time[::-1]  # x, then x reversed
    y_upper = list(ts_pred + ts_error)
    y_lower = list(ts_pred - ts_error)
    e_y = y_upper + y_lower[::-1]  # upper, then lower reversed
    cont_error_bars = go.Scatter(
        x=e_x,
        y=e_y,
        fill="toself",
        fillcolor="rgba(0,255,255,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=False,
    )
    fig.add_trace(cont_error_bars)

    fig.update_layout(
        height=510, margin=dict(l=80, r=80, b=10, t=40, pad=0), paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.data[0].update(mode="markers+lines")

    return fig


@app.callback(Output("interpret-figure", "figure"), [Input("patient-dropdown", "value")])
def update_ts_pred_interpret_figure(patient_id):
    print("update_ts_pred_interpret_figure()...")

    problem_bundle = problem_bundles["time_series_prediction"]
    patient_row_idx = data_bundle.testing_id_row_map[patient_id]

    label_name = problem_bundle.extra_settings.metric_parameters["label_name"]
    temporal_features = problem_bundle.results.dataset_training.feature_name["temporal"]
    static_features = problem_bundle.results.dataset_training.feature_name["static"]

    interpretations = problem_bundle.results.test_s_hat

    if problem_bundle.extra_settings.model_parameters["static_mode"] == "concatenate":
        temporal_features = temporal_features + static_features
    if problem_bundle.extra_settings.model_parameters["time_mode"] == "concatenate":
        temporal_features = temporal_features + ["time"]
    if label_name[0] in temporal_features:
        temporal_features = list_diff(temporal_features, label_name)

    fig = px.imshow(
        img=np.transpose(interpretations[patient_row_idx, :, :]),
        title="Feature Importance",
        labels=dict(x="Hours", y="Features", color="Importance"),
        x=np.arange(interpretations.shape[1]),
        y=temporal_features,
        template="plotly_dark",
        range_color=(0.0, 1.0),
    )

    fig.update_yaxes(tickfont=dict(size=9), dtick=1)
    fig.update_layout(
        height=510, margin=dict(l=80, r=80, b=10, t=40, pad=0), paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


@app.callback(Output("as-figure", "figure"), [Input("patient-dropdown", "value")])
def update_as_figure(patient_id):
    print("update_as_figure()...")

    problem_bundle = problem_bundles["active_sensing"]
    patient_row_idx = data_bundle.testing_id_row_map[patient_id]

    label_name = problem_bundle.extra_settings.metric_parameters["label_name"]
    temporal_features = problem_bundle.results.dataset_training.feature_name["temporal"]
    static_features = problem_bundle.results.dataset_training.feature_name["static"]

    recommendations = problem_bundle.results.test_s_hat[patient_row_idx]
    recommendations_nonzero_bool = recommendations.sum(axis=0) > 0.0
    recommendations_nonzero_idx = (recommendations_nonzero_bool).nonzero()
    recommendations_nonzero = recommendations[:, recommendations_nonzero_bool]

    if problem_bundle.extra_settings.model_parameters["static_mode"] == "concatenate":
        temporal_features = temporal_features + static_features
    if problem_bundle.extra_settings.model_parameters["time_mode"] == "concatenate":
        temporal_features = temporal_features + ["time"]
    if label_name[0] in temporal_features:
        temporal_features = list_diff(temporal_features, label_name)

    temporal_features = np.array(temporal_features)[recommendations_nonzero_idx]

    # Grid component.
    data_mx = np.transpose(recommendations_nonzero)
    x_vals = np.arange(recommendations_nonzero.shape[0])
    fig = px.imshow(
        img=data_mx,
        title="Recommended Measurements",
        labels=dict(x="Hours Ahead", y="Features", color="Importance"),
        x=x_vals,
        y=temporal_features,
        template="plotly_dark",
        aspect="equal",
        color_continuous_scale=["rgb(0,0,102)", "rgb(128,128,255)"],
    )

    # Cross markers component.
    m_y, m_x = (data_mx > 0.0).nonzero()
    m_y = [temporal_features[val] for val in m_y]
    crosses = go.Scatter(
        x=m_x,
        y=m_y,
        mode="markers",
        showlegend=False,
        marker=dict(symbol="x", opacity=1.0, color="white", size=12, line=dict(width=1),),
        name="Measure",
    )
    fig.add_trace(crosses)

    adjust_max = 0.5
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", xaxis_range=(min(x_vals), max(x_vals) + adjust_max), coloraxis_showscale=False
    )

    return fig
