"""Interpretation model define.
"""

# Necessary modules
from interpretation import TInvase


def interpretation(interpretation_model_name, model_parameters, pred_class, task):
    """Determine interpretation model.
    
    Args:
        - interpretation_model_name: 'tinvase'
        - model_parameters: parameters of the interpretor models
        - pred_class: predictor
        - task: 'classification' or 'regression':
            
    Returns:
        - interpretor: determined interpretation model
    """
    assert interpretation_model_name in ["tinvase"]
    # Set class
    if interpretation_model_name == "tinvase":
        interpretor = TInvase(predictor_model=pred_class, task=task)
    # Set parameters
    interpretor.set_params(**model_parameters)

    return interpretor
