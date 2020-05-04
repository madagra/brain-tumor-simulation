import os
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# plotly identifiers and constants
SLIDER_ID = "my-slider-id"
PLOT_ID = "my-plot-id"
PLOT_CONTAINER_ID = "my-plot-container-id"
SLIDER_STEPS = 15
DEFAULT_COLORSCALE = "geyser"

# filename for the solution
solution_dir = "solution"
solution_prefix = "slice"
brain_fname = os.path.join(solution_dir, "diffusion_slice.csv")

# text available on the interface
description = """
## Brain tumor diffusion simulation

This simple web application allows to visualize the evolution of a brain tumor according to simple diffusion 
equations. See [here](www.google.com) for more explanation on the underlying equation and the full code used 
to produce the solution.
"""

slider_instructions = """
Use the slider to see how the tumor evolves over time to cover larger parts of the brain.
"""

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def slider():
    """
    Build the slider depending on how many files are available in the
    folder containing the solution of the equation for the brain tumor
    diffusion

    Returns
    -------
    A dcc.Slider object with the right steps and ticks
    """

    # assuming that there are no other directories inside solution
    for _, _, files in os.walk(solution_dir):
        n_files = len(files)
    assert n_files is not None

    if n_files-1 > SLIDER_STEPS:
        step = n_files // SLIDER_STEPS + 1
        marks = { n:f"t_{n} " for n in range(0, n_files-1, step) }
    else:
        marks = { n:f"t_{n} " for n in range(0, n_files-1) }

    return dcc.Slider(
                id=SLIDER_ID, 
                min=0, 
                max=n_files-1, 
                marks=marks,
                step=None)


def get_contour_data(fname):
    """
    Preprocess the raw diffusion data to obtain a vector which can be
    consumed by the Plotly Contour API

    Parameters
    ----------
    fname: the file name to read the raw data from

    Returns
    -------
    A numpy 2D array with the data points for the contour plot
    """

    slice_df = pd.read_csv(fname)

    nx = len(np.unique(np.array(slice_df["x"])))
    ny = len(np.unique(np.array(slice_df["y"])))
    z = np.array(slice_df["value"])

    data = []
    for i in range(ny):
        data.append(z[(i*nx):((i+1)*nx)])
    return np.array(data)


def plot_slice(fname):
    """
    Plot a 2D contour plot of a slice of a 3D array

    Parameters
    ----------
    fname: str with the filename containing the array slice with the tumor diffusion coefficients

    Returns
    -------
    plotly Figure holding the contour plot
    """

    titles = ["2D Slice Tumor Evolution", "2D Slice Normal Brain"]
    specs=[[{'type': 'contour'}, {'type': 'contour'}]]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=titles, specs=specs)

    data = get_contour_data(fname)    
    fig1 = go.Contour(
                z=data,
                colorscale=DEFAULT_COLORSCALE,
                visible=True,
                coloraxis = "coloraxis",
                showlegend=False
            )

    data = get_contour_data(brain_fname)
    fig2 = go.Contour(
            z=data,
            colorscale=DEFAULT_COLORSCALE,
            coloraxis = "coloraxis",
            visible=True
        )

    # fig = go.Figure()
    fig.add_trace(fig1, row=1, col=1)
    fig.add_trace(fig2, row=1, col=2)
    fig.update_layout(width=1500, height=750, autosize=False, coloraxis = {'colorscale':DEFAULT_COLORSCALE})

    return dcc.Graph(
                id=PLOT_ID, 
                figure=fig
            )


@app.callback(
    Output(component_id=PLOT_CONTAINER_ID, component_property='children'),
    [Input(component_id=SLIDER_ID, component_property='value')]
)
def update_graph(slider_value):
    """
    Update the graph depending on which value of the slider is selected

    Parameters
    ----------
    slider_value: the current slider value, None when starting the app

    Returns
    -------
    dcc.Graph object holding the new Plotly Figure as given by the plot_slice function
    """
    value = slider_value if slider_value is not None else 0
    fname = os.path.join(solution_dir, f"{solution_prefix}_{value}.csv")
    return plot_slice(fname)


# build the application
app.layout = html.Div(children=[
    html.Div(dcc.Markdown(description)),
    html.Hr(),
    html.Center([
        dcc.Markdown(slider_instructions),
        html.Div([ 
            slider() 
            ], style={"width": "50%"}
        ),
        html.Div(id=PLOT_CONTAINER_ID)
    ]),
    html.Hr()
])


if __name__ == '__main__':
    app.run_server(debug=True)
