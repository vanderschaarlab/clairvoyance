"""Define predictive models.

Possible models: 'attention', 'rnn', 'lstm', 'gru', 'tcn', 'transformer'
"""

# Necessary modules
from prediction import GeneralRNN, Attention, TemporalCNN, TransformerPredictor, GBM


def prediction(model_name, model_parameters, task):
    """Determine predictive model.
    
    Args:
        - model_name: 'attention', 'rnn', 'lstm', 'gru', 'tcn', 'transformer'
        - model_parameters: parameters of the predictive models
        - task: 'classification' or 'regression':
            
    Returns:
        - pred_class: predictive model
    """
    assert model_name in ["attention", "rnn", "lstm", "gru", "tcn", "transformer", "gbm"]

    # Set model parameters
    if model_name in ["attention", "rnn", "lstm", "gru", "tcn"]:
        if "n_head" in model_parameters.keys():
            del model_parameters["n_head"]

    if model_name in ["tcn", "transformer"]:
        if "model_type" in model_parameters.keys():
            del model_parameters["model_type"]

    # Set predictive model
    if model_name == "attention":
        pred_class = Attention(task=task)
        pred_class.set_params(**model_parameters)

    elif model_name in ["rnn", "lstm", "gru"]:
        pred_class = GeneralRNN(task=task)
        pred_class.set_params(**model_parameters)

    elif model_name == "tcn":
        pred_class = TemporalCNN(task=task)
        pred_class.set_params(**model_parameters)

    elif model_name == "transformer":
        pred_class = TransformerPredictor(task=task)
        pred_class.set_params(**model_parameters)
    elif model_name == "gbm":
        pred_class = GBM(task=task)
        pred_class.set_params(**model_parameters)

    return pred_class
