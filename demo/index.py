"""
Clairvoyance demo.
"""
import warnings

warnings.filterwarnings("ignore")

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import app_mimic_main, app_mimic_antibiotics  # pylint: disable=unused-import


# Pages.
page_info = {
    "mimic-main": {"path": "/mimic-main", "nav-text": "Critical Care: A", "nav-id": "mimic-main-link",},
    "mimic-antibiotics": {
        "path": "/mimic-antibiotics",
        "nav-text": "Critical Care: B",
        "nav-id": "mimic-antibiotics-link",
    },
}

# Set up the layout.
sidebar = html.Div(
    [
        html.Img(src=app.get_asset_url("clairvoyance_logo_white.png"), id="clairvoyance-logo"),
        html.P(
            "A unified, end-to-end AutoML pipeline for medical time series", className="lead", id="clairvoyance-lead",
        ),
        html.Hr(),
        html.P("Data Source:"),
        dbc.Nav(
            [dbc.NavLink(pi["nav-text"], href=pi["path"], id=pi["nav-id"]) for pi in page_info.values()],
            vertical=True,
            pills=True,
        ),
    ],
    id="clairvoyance-sidebar",
)
content = html.Div(id="page-content")
location = dcc.Location(id="url")
app.layout = html.Div([location, sidebar, content])

# Validation layout (must include components from all pages).
app.validation_layout = html.Div(
    [location, sidebar, content, app_mimic_main.validation_layout, app_mimic_antibiotics.layout,]
)


# Enable the active sidebar nav item based on URL request.
@app.callback(
    [Output(pi["nav-id"], "active") for pi in page_info.values()], [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        initial_page_active_status = [False] * len(page_info)
        initial_page_active_status[0] = True
        return initial_page_active_status
    return [pathname == pi["path"] for pi in page_info.values()]


@app.callback(
    Output("page-content", "children"), [Input("url", "pathname")],
)
def render_page_content(pathname):

    if pathname in ["/", page_info["mimic-main"]["path"]]:
        return app_mimic_main.layout
    elif pathname == page_info["mimic-antibiotics"]["path"]:
        return app_mimic_antibiotics.layout

    # 404.
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_hot_reload=False, dev_tools_ui=True)
