"""Output visualizations.

- In Table for one-shot prediction setting
- In Graph for online prediction setting
- In Heatmap for interpretation

(1) Print performance 
(2) Print predictions for particular patient
(3) Print predictions + uncertainty for particular patient
(4) Print interpretations of the particular prediction
(5) Print counterfactual predictions
"""

# Necessary packages
from texttable import Texttable
import matplotlib.pyplot as plt
import numpy as np
from utils import list_diff


def print_performance(performance, metric_sets, metric_parameters):
    """Visualize the overall performance.
    
    Args:
        - performance: dictionary based performances
        - metric_sets: sets of metric
        - metric_parameters: parameters for the problem and labels
        
    Returns:        
        - For online prediction, returns graphs
        - For one-shot prediction, returns table
    """
    # Parameters
    label_sets = metric_parameters["label_name"]
    problem = metric_parameters["problem"]
    graph_format = ["bo-", "r+--", "gs-.", "cp:", "m*-"]

    # For one-shot prediction setting
    if problem == "one-shot":
        # Initialize table
        perf_table = Texttable()
        first_row = ["metric/label"] + label_sets
        perf_table.set_cols_align(["c" for _ in range(len(first_row))])
        multi_rows = [first_row]

        # For each metric
        for i in range(len(metric_sets)):
            metric_name = metric_sets[i]
            curr_row = [metric_name]

            # For each label
            for j in range(len(label_sets)):
                label_name = label_sets[j]
                curr_key = label_name + " + " + metric_name
                curr_row = curr_row + [str(performance[curr_key])]
                multi_rows = multi_rows + [curr_row]

        perf_table.add_rows(multi_rows)
        # Print table
        print(perf_table.draw())
        # Return table
        return perf_table.draw()

    # For online prediction setting
    elif problem == "online":
        # Initialize the graph
        figs = []

        # For each metric
        for i in range(len(metric_sets)):
            metric_name = metric_sets[i]
            curr_row = [metric_name]

            fig = plt.figure(i, figsize=(8, 5))
            legend_set = []

            # For each label
            for j in range(len(label_sets)):
                label_name = label_sets[j]
                curr_key = label_name + " + " + metric_name
                curr_row = curr_row + [str(performance[curr_key])]
                curr_perf = performance[curr_key]
                legend_set = legend_set + [label_name]

                plt.plot(range(len(curr_perf) - 1), curr_perf[:-1], graph_format[j])
                plt.xlabel("Sequence Length", fontsize=10)
                plt.ylabel("Performance", fontsize=10)
                plt.legend(legend_set, fontsize=10)
                plt.title("Performance metric: " + metric_name, fontsize=10)
                plt.grid()
                # Print figure
                plt.show()

            fig.patch.set_facecolor("#f0f2f6")
            figs.append(fig)
        # Return figure
        return figs


def print_prediction(predictions, metric_parameters):
    """Visualize the predictions.
    
    Args:
        - predictions: predictions of each patient
        - metric_parameters: parameters for the problem and labels
        
    Returns:        
        - For online predictions, returns graphs
        - For one-shot predictions, returns table
    """
    # Parameters
    label_sets = metric_parameters["label_name"]
    problem = metric_parameters["problem"]
    graph_format = ["bo-", "r+--", "gs-.", "cp:", "m*-"]

    # For one-shot prediction setting
    if problem == "one-shot":
        # Initialize table
        perf_table = Texttable()
        first_row = ["id/label"] + label_sets
        perf_table.set_cols_align(["c" for _ in range(len(first_row))])
        multi_rows = [first_row]

        for i in range(predictions.shape[0]):
            curr_row = [str(i + 1)]

            # For each label
            for j in range(len(label_sets)):
                label_name = label_sets[j]
                curr_row = curr_row + [predictions[i, j]]

            multi_rows = multi_rows + [curr_row]

        perf_table.add_rows(multi_rows)
        # Print table
        print(perf_table.draw())
        # Return table
        return perf_table.draw()

    # For online prediction setting
    elif problem == "online":
        # Initialize graph
        figs = []

        for i in range(predictions.shape[0]):
            fig = plt.figure(i + 10, figsize=(8, 5))
            legend_set = []

            # For each label
            for j in range(len(label_sets)):
                label_name = label_sets[j]
                curr_perf = predictions[i][:, j]
                legend_set = legend_set + [label_name]
                plt.plot(range(len(curr_perf) - 1), curr_perf[:-1], graph_format[j])

            plt.xlabel("Sequence Length", fontsize=10)
            plt.ylabel("Predictions", fontsize=10)
            plt.legend(legend_set, fontsize=10)
            plt.title("ID: " + str(i + 1), fontsize=10)
            plt.grid()
            # Print graph
            plt.show()

            fig.patch.set_facecolor("#f0f2f6")
            figs.append(fig)
        # Return graph
        return figs


def print_uncertainty(predictions, uncertainties, metric_parameters):
    """Visualize the predictions with uncertainties.
    
    Args:
        - predictions: predictions of each patient
        - uncertainties: uncertainties of each prediction
        - metric_parameters: parameters for the problem and labels
        
    Returns:        
        - For online predictions, returns graphs
        - For one-shot predictions, returns table
    """
    # Parameters
    label_sets = metric_parameters["label_name"]
    problem = metric_parameters["problem"]
    graph_format = ["bo-", "r+--", "gs-.", "cp:", "m*-"]

    # For one-shot prediction setting
    if problem == "one-shot":
        # Initialize table
        perf_table = Texttable()
        first_row = ["id/label"] + label_sets
        perf_table.set_cols_align(["c" for _ in range(len(first_row))])
        multi_rows = [first_row]

        for i in range(predictions.shape[0]):
            curr_row = [str(i + 1)]

            # For each label
            for j in range(len(label_sets)):
                label_name = label_sets[j]
                curr_row = curr_row + [
                    str(np.round(predictions[i, j], 4)) + "+-" + str(np.round(uncertainties[i, j], 4))
                ]
            multi_rows = multi_rows + [curr_row]

        perf_table.add_rows(multi_rows)
        # Print table
        print(perf_table.draw())
        # Return table
        return perf_table.draw()

    # For online prediction setting
    elif problem == "online":
        # Initialize the graph
        figs = []

        for i in range(predictions.shape[0]):
            fig = plt.figure(i + 10, figsize=(8, 5))
            legend_set = []

            # For each label
            for j in range(len(label_sets)):
                label_name = label_sets[j]

                curr_perf = predictions[i][:, j]
                under_line = curr_perf - uncertainties[i][:, j]
                over_line = curr_perf + uncertainties[i][:, j]
                legend_set = legend_set + [label_name]
                plt.plot(range(len(curr_perf) - 1), curr_perf[:-1], graph_format[j])
                plt.fill_between(range(len(curr_perf) - 1), under_line[:-1], over_line[:-1], alpha=0.5)

            plt.xlabel("Sequence Length", fontsize=10)
            plt.ylabel("Predictions", fontsize=10)
            plt.legend(legend_set, fontsize=10)
            plt.title("ID: " + str(i + 1), fontsize=10)
            plt.grid()
            # Print graph
            plt.show()

            fig.patch.set_facecolor("#f0f2f6")
            figs.append(fig)
        # Return graph
        return figs


def print_interpretation(interpretations, feature_name, metric_parameters, model_parameters):
    """Visualize the interpretations.
    
    Args:
        - interpretations: interpretations of each patient
        - temporal features: y-axis of the heatmap
        - metric_parameters: parameters for the problem and labels
        - model_parameters: parameters for the predictor model (concatenation)
        
    Returns:        
        - Feature and temporal importance for each patient on heatmap
    """
    label_name = metric_parameters["label_name"]

    # Define feature name
    temporal_features = feature_name["temporal"]
    static_features = feature_name["static"]

    if model_parameters["static_mode"] == "concatenate":
        temporal_features = temporal_features + static_features
    if model_parameters["time_mode"] == "concatenate":
        temporal_features = temporal_features + ["time"]
    if label_name[0] in temporal_features:
        temporal_features = list_diff(temporal_features, label_name)

    figs = []

    # Generate heatmap
    for i in range(interpretations.shape[0]):
        fig = plt.figure(figsize=(8, 10))
        plt.imshow(np.transpose(interpretations[i, :, :]), cmap="Greys_r")
        plt.xticks(np.arange(interpretations.shape[1]))
        plt.yticks(np.arange(interpretations.shape[2]), temporal_features)
        plt.colorbar()
        plt.clim(0, 1)
        plt.xlabel("Sequence Length", fontsize=10)
        plt.ylabel("Features", fontsize=10)
        plt.title("Feature and temporal importance for patient ID: " + str(i + 1), fontsize=10)
        plt.show()

        fig.patch.set_facecolor("#f0f2f6")
        figs.append(fig)

    return figs


def print_counterfactual_predictions(patient_history, treatment_options, counterfactual_predictions):
    """Visualize the counterfactual predictions.
    
    Args:
        - patient_history
        - treatment_options
        - counterfactual_predictions
            
    Returns:
        - Counterfactual predictions in graph
    """
    prediction_horizon = treatment_options.shape[1]
    history_length = patient_history.shape[0]

    figs = []
    fig = plt.figure(10, figsize=(8, 4))

    plt.plot(range(history_length), patient_history, label="Patient history", color="#237F57")
    plt.axvline(x=history_length - 1, linestyle="--")
    for (index, counterfactual) in enumerate(counterfactual_predictions):
        extended_counterfactual = np.concatenate([[patient_history[-1]], counterfactual])
        plt.plot(range(history_length - 1, history_length + prediction_horizon), extended_counterfactual)

        no_treatment_idx = np.where(treatment_options[index] == 0)[0]
        plt.scatter(
            np.array(range(history_length, history_length + prediction_horizon))[no_treatment_idx],
            counterfactual[no_treatment_idx],
            marker="o",
            facecolors="none",
            s=150,
            c="#3788CF",
            Label="No treatment",
        )

        treatment_idx = np.where(treatment_options[index] == 1)[0]
        plt.scatter(
            np.array(range(history_length, history_length + prediction_horizon))[treatment_idx],
            counterfactual[treatment_idx],
            marker="x",
            s=150,
            c="#C93819",
            Label="Treatment",
        )

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.xlabel("Timestep", fontsize=15)
    plt.ylabel("Predictions", fontsize=15)
    plt.title("Counterfactual predictions", fontsize=15)
    plt.show()

    fig.patch.set_facecolor("#f0f2f6")
    figs.append(fig)

    return fig
