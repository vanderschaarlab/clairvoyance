"""Uncertainty estimation model define.
"""

# Necessary modules
from uncertainty import EnsembleUncertainty


def uncertainty(uncertainty_model_name, model_parameters, pred_class, task):
    """Determine uncertainty model.
    
    Args:
        - uncertainty_model_name: 'ensemble'
        - model_parameters: parameters of the interpretor models
        - pred_class: predictor
        - task: 'classification' or 'regression':
            
    Returns:
        - uncertainty_model: determined uncertainty estimation model
    """
    assert uncertainty_model_name in ["ensemble"]

    if uncertainty_model_name == "ensemble":
        uncertainty_model = EnsembleUncertainty(
            predictor_model=pred_class, ensemble_model_type=["rnn", "lstm", "gru"], task=task
        )

    uncertainty_model.set_params(**model_parameters)

    return uncertainty_model
