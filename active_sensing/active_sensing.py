"""Define Active sensing models.

(1) ASAC: RNN based model with active-sensing framework.
(2) DeepSensing: RNN based model with supervised learning framework.
"""

# Necessary modules
from active_sensing import Asac, DeepSensing


def active_sensing(active_sensing_model_name, model_parameters, task):
    """Determine active sensing model.
    
    Args:
        - active_sensing_model_name: 'asac', 'deepsensing'
        - model_parameters: parameters of the active sensing models
        - task: 'classification' or 'regression':
            
    Returns:
        - active_sensing_class: active sensing model
    """
    assert active_sensing_model_name in ["asac", "deepsensing"]

    # Define active sensing model
    if active_sensing_model_name == "asac":
        active_sensing_class = Asac(task=task)

    elif active_sensing_model_name == "deepsensing":
        active_sensing_class = DeepSensing(task=task)

    # Set parameters
    active_sensing_class.set_params(**model_parameters)

    return active_sensing_class
