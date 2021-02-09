"""
Clairvoyance demo.
"""
import os
import sys

clairvoyance_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if clairvoyance_root not in sys.path:
    print(f"Adding clairvoyance path to PYTHONPATH: '{clairvoyance_root}'")
    sys.path.append(clairvoyance_root)

import warnings

warnings.filterwarnings("ignore")

import dash
import dash_bootstrap_components as dbc

from backend import load_backend


working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "wd"))
print(f"Demo working directory at: '{working_dir}'")
if not os.path.exists(working_dir):
    os.makedirs(working_dir)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "clairvoyance"
server = app.server
backend_contents = load_backend(working_dir=working_dir)


# Define the subsets of content for demo purposes.
PATIENT_IDS = [200010, 200035, 200040]
DUMMY_MAP = {200010: 0, 200035: 1, 200040: 2}  # TODO: Deal with this!
