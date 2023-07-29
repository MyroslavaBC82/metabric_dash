import dash
from dash import dcc, html, dash_table, callback
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import Isomap, LocallyLinearEmbedding
import plotly.graph_objs as go
import os
from uuid import uuid4
import diskcache
from diskcache import Cache

# Set up diskcache for caching
cache = diskcache.Cache("./cache")

# Caching duration (e.g., 60 seconds = 1 minute)
CACHE_DURATION = 180

dash.register_page(__name__)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Load the dataset
data = pd.read_csv("METABRIC_RNA_Mutation4.csv")
# file_path = '/home/MyroslavaBC82/metabric_dash/METABRIC_RNA_Mutation4.csv'
# data = pd.read_csv(file_path)
available_variables = list(data.columns)
EPSILON = 1e-9 
numerical_data = data.select_dtypes(include='number')


def linear_layout_algorithm(data, iterations=1000, learning_rate=0.1, Vmax=5, Smax=10):
    N, D = data.shape

    # Initialize positions and velocities randomly
    positions = np.random.random((N, 2))
    velocities = np.zeros((N, 2))

    for iteration in range(iterations):
        # Update neighbor sets
        neighbor_sets = []
        for i in range(N):
            distances = np.linalg.norm(data - data[i], axis=1)
            indices = np.argsort(distances)[1:Vmax+1]  # Exclude self
            neighbor_sets.append(indices)

        # Update positions and velocities
        for i in range(N):
            forces = np.zeros(2)
            for j in neighbor_sets[i]:
                dist_ij = np.linalg.norm(positions[j] - positions[i])
                dist_ij_high = np.linalg.norm(data[j] - data[i])
                force_ij = dist_ij - dist_ij_high
                if np.isfinite(force_ij):
                    forces += force_ij * (positions[j] - positions[i]) * np.divide(1, (dist_ij + EPSILON), where=(dist_ij > EPSILON))

            subset = np.random.choice(N, Smax, replace=False)
            for j in subset:
                dist_ij = np.linalg.norm(positions[j] - positions[i])
                dist_ij_high = np.linalg.norm(data[j] - data[i])
                force_ij = dist_ij - dist_ij_high
                if np.isfinite(force_ij):
                    forces += force_ij * (positions[j] - positions[i]) * np.divide(1, (dist_ij + EPSILON), where=(dist_ij > EPSILON))

            max_force_magnitude = np.max(np.linalg.norm(forces))  # Calculate norm without specifying axis
            normalized_forces = forces / max_force_magnitude  # Normalize forces
            velocities[i] += learning_rate * normalized_forces  # Update velocity with learning_rate
            positions[i] += velocities[i]  # Update position

        positions -= np.mean(positions, axis=0)  # Normalize positions
        std = np.std(positions, axis=0)
        positions = positions / std[np.newaxis, :] if not np.allclose(std, 0) else positions

    return positions



layout = html.Div(
    children=[
            html.Div(
                dcc.Loading(
                    type="circle",
                    children=[html.Div("Loading...", style={"font-size": "20px"})],
                    fullscreen=True,
                ),
                id="loading-spinner",
            ),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="six columns", style={'background-color': 'white', 'padding': '10px', 'box-shadow': '2px 2px 10px rgba(0, 0, 0, 0.2)'},
                    children=[
                        html.Div(
                            children=[
                                  # Add loading to the graph
                                    dcc.Graph(
                                        id="cluster-graph",
                                        hoverData={"points": [{"hovertemplate": ""}]},
                                    ),
                                    
                            ]
                        ),
                    ],
                ),
                html.Div(
                    className="six columns",
                    children=[
                        html.Div(
                            children=[
                                html.Label("Method:"),
                                dcc.Dropdown(
                                    id="method-dropdown",
                                    options=[
                                        {"label": "UMAP", "value": "umap"},
                                        {"label": "t-SNE", "value": "tsne"},
                                        {"label": "PCA", "value": "pca"},
                                        {"label": "Isomap", "value": "isomap"},
                                        {"label": "LLE", "value": "lle"},
                                        {
                                            "label": "Linear Layout Algorithm",
                                            "value": "linear_layout_algorithm",
                                        },
                                    ],
                                    value="umap",
                                    style={
                                        "width": "150px",
                                        "display": "inline-block",
                                    },
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.Label("Number of Components:"),
                                dcc.Input(
                                    id="n-components-input",
                                    type="number",
                                    min=2,
                                    max=len(available_variables),
                                    value=3,
                                    step=1,
                                    style={
                                        "width": "100px",
                                        "display": "inline-block",
                                        "opacity": 1,  # Full opacity by default
                                    },
                                ),
                            ],
                            id="n-components-container",
                        ),
                        html.Div(
                            children=[
                                html.Label("Number of Neighbours:"),
                                dcc.Input(
                                    id="n-neighbours-input",
                                    type="number",
                                    min=2,
                                    max=len(available_variables),
                                    value=5,
                                    step=1,
                                    style={
                                        "width": "100px",
                                        "display": "inline-block",
                                        "opacity": 1,  # Full opacity by default
                                    },
                                ),
                            ],
                            id="n-neighbours-container",
                        ),
                        html.Div(
                            children=[
                                html.Label("Select Color Variable:"),
                                dcc.Dropdown(
                                    id="color-variable-input",
                                    options=[
                                        {
                                            "label": variable,
                                            "value": variable,
                                        }
                                        for variable in available_variables[:32]
                                    ],
                                    value="chemotherapy",
                                    style={
                                        "width": "150px",
                                        "display": "inline-block",
                                    },
                                ),
                            ]
                        ),
                        html.Div(
                            className="row",
                            children=[
                                html.Div(
                                    className="six columns",
                                    children=[
                                        html.Label(
                                            "Select Variables for Clustering Graph:"
                                        ),
                                        html.Div(
                                            className="scrollable",
                                            style={
                                                "height": "230px",
                                                "overflowY": "scroll",
                                                "border": "1px solid #ccc",
                                                "border-radius": "5px",
                                                "padding": "5px",
                                            },
                                            children=[
                                                dcc.Checklist(
                                                    id="variable-checkboxes",
                                                    options=[
                                                        {
                                                            "label": variable,
                                                            "value": variable,
                                                        }
                                                        for variable in numerical_data
                                                    ],
                                                    value=[
                                                        "brca1",
                                                        "brca2",
                                                        "palb2",
                                                        "pten",
                                                        "tp53",
                                                        "atm",
                                                        "cdh1",
                                                        "chek2",
                                                        "nbn",
                                                        "nf1",
                                                        "stk11",
                                                        "bard1",
                                                        "mlh1",
                                                        "msh2",
                                                        "msh6",
                                                        "pms2",
                                                        "epcam",
                                                        "rad51c",
                                                        "rad51d",
                                                        "rad50",
                                                        "rb1",
                                                        "rbl1",
                                                        "rbl2",
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="six columns",
                                    children=[
                                        html.Label(
                                            "Select Variables for Parallel Coordinates:"
                                        ),
                                        html.Div(
                                            className="scrollable",
                                            style={
                                                "height": "230px",
                                                "overflowY": "scroll",
                                                "border": "1px solid #ccc",
                                                "border-radius": "5px",
                                                "padding": "5px",
                                            },
                                            children=[
                                                dcc.Checklist(
                                                    id="show-variables-checkboxes",
                                                    options=[
                                                        {
                                                            "label": variable,
                                                            "value": variable,
                                                        }
                                                        for variable in numerical_data
                                                    ],
                                                    value=[
                                                        "brca1",
                                                        "brca2",
                                                        "palb2",
                                                        "pten",
                                                        "tp53",
                                                        "atm",
                                                        "cdh1",
                                                        "chek2",
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="row", style={'margin-top': '20px', 'background-color': 'white', 'padding': '10px', 'box-shadow': '2px 2px 10px rgba(0, 0, 0, 0.2)'},
            children=[
                html.Div(
                    className="twelve columns",
                    children=[
                        html.Div(
                            children=[
                                html.Label("Parallel Coordinates graph:"),
                                
                                    dcc.Graph(id="parallel-coordinates-graph"),
                                    
                            ]
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="row", style={'margin-top': '20px','background-color': 'white', 'padding': '10px', 'box-shadow': '2px 2px 10px rgba(0, 0, 0, 0.2)'},
            children=[
                html.Div(
                    style={"width": "100%"},
                    children=[
                        html.H4("Data Table"),
                        dash_table.DataTable(
                            id="data-table",
                            columns=[
                                {"name": col, "id": col} for col in data.columns
                            ],
                            data=data.to_dict("records"),
                            style_table={
                                "height": "300px",
                                "overflowY": "scroll",
                            },
                            style_cell={"padding": "3px"},
                            style_header={
                                "backgroundColor": "rgb(230, 230, 230)",
                                "fontWeight": "bold",
                            },
                            row_selectable="multi",
                            selected_rows=[],
                        ),
                    ],
                )
            ],
        ),
    ],
    style={
        "font-family": "Arial, sans-serif",
        "margin": "20px",
        "padding": "20px",
        "background-color": "#F5F5F5",
    },
)

@callback(Output("loading-spinner", "style"), Input("data-table", "data"))
def hide_loading_spinner(data):
    if data is not None:
        return {"display": "none"}
    else:
        return {"display": "block"}

# Callback to adjust the opacity/transparency based on the selected method
@callback(
    [
        dash.dependencies.Output("n-components-container", "style"),
        dash.dependencies.Output("n-neighbours-container", "style"),
    ],
    [dash.dependencies.Input("method-dropdown", "value")]
)
def adjust_opacity_based_on_method(method):
    if method in ["umap", "isomap", "lle"]:
        return {"opacity": 1}, {"opacity": 1}
    elif method == "tsne":
        return {"opacity": 1}, {"opacity": 0.5}
    else:  # For PCA and Linear Layout Algorithm
        return {"opacity": 0.5}, {"opacity": 0.5}


@callback(
    Output("parallel-coordinates-graph", "figure"),
    [Input("show-variables-checkboxes", "value"),
     Input("data-table", "selected_rows")],
)
def update_parallel_coordinates(selected_show_variables, selected_rows):
    df_selected_variables = data[selected_show_variables]

    # Create a copy of the data for plotting
    plot_data = df_selected_variables.copy()

    # Initialize the color column for all rows to 0 (unselected)
    plot_data["color"] = 0

    # Highlight selected rows with 1
    if selected_rows:
        selected_indices = [data.iloc[row].name for row in selected_rows]
        plot_data.loc[selected_indices, "color"] = 1

    fig = go.Figure(data=go.Parcoords(
        line=dict(color=plot_data["color"],  # Color by the 'color' column
                  colorscale=[[0, '#EAEDED'], [1, 'red']],  # Colorscale: gray for unselected, red for selected
                  showscale=False),  # Show color scale bar
        dimensions=[
            dict(range=[min(plot_data[col]), max(plot_data[col])],
                 label=col,
                 values=plot_data[col])
            for col in selected_show_variables]
    ))

    if selected_rows:
        selected_indices = [
            data.iloc[row].name for row in selected_rows
        ]
        fig.update_traces(
            line=dict(color="red"), selector={"line.color": "gray"}  # Highlight selected rows in red
        )

    return fig


@callback(
    [Output("cluster-graph", "figure"),
     Output("data-table", "selected_rows")],
    [
        Input("variable-checkboxes", "value"),
        Input("method-dropdown", "value"),
        Input("n-components-input", "value"),
        Input("n-neighbours-input", "value"),
        Input("color-variable-input", "value"),
        Input("data-table", "derived_virtual_data"),
        Input("data-table", "selected_rows"),
        Input("cluster-graph", "clickData"),
    ],
    [State("cluster-graph", "figure"),
     State("data-table", "selected_rows")],
)
def update_cluster_graph(
    selected_variables,
    method,
    n_components,
    n_neighbours,
    color_variable,
    derived_virtual_data,
    selected_rows_data_table,
    click_data,
    current_figure,
    current_selected_rows,
):
    
    # Generate a unique key for this callback's cache entry
    cache_key = (
        selected_variables,
        method,
        n_components,
        n_neighbours,
        color_variable,
        derived_virtual_data,
        selected_rows_data_table,
        click_data,
        current_figure,
        current_selected_rows,
    )

    # Check if the result is already cached and return it if available
    if cache_key in cache:
        cached_result = cache[cache_key]
        return cached_result
    # Perform dimensionality reduction based on the selected method
    if derived_virtual_data is not None:
        data_subset = pd.DataFrame(derived_virtual_data)
        selected_data = data_subset[selected_variables]
    else:
        selected_data = data[selected_variables]

    if method == "umap":
        reducer = umap.UMAP(
            n_neighbors=n_neighbours, min_dist=0.3, n_components=n_components, random_state=42
        )
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=42)
    elif method == "pca":
        reducer =  PCA(
            n_components=min(n_components, len(selected_variables)),
            svd_solver="randomized",
            random_state=42,
        )
    elif method == "isomap":
        reducer =  Isomap(n_neighbors=n_neighbours, n_components=n_components)
    elif method == "lle":
        reducer =  LocallyLinearEmbedding(n_neighbors=n_neighbours, n_components=n_components)
    elif method == "linear_layout_algorithm":
        reducer =  None 

    if reducer:
        embedding = reducer.fit_transform(selected_data)
        # Plot the scatter plot with different colors for each variable
        fig = px.scatter(
            data_frame=data,
            x=embedding[:, 0],
            y=embedding[:, 1],
            color=color_variable,
            title="Clustering",
            hover_data={"patient_id": True},
        )
    else:
        # Apply linear layout algorithm
        positions = linear_layout_algorithm(selected_data.values)
        df = pd.DataFrame(selected_data, columns=selected_variables)
        df["x"] = positions[:, 0]
        df["y"] = positions[:, 1]
        df['color_variable'] = data[color_variable]  # Assign the color variable to the DataFrame
        df['patient_id'] = data['patient_id']  # Include the 'patient_id' column in df

        # Plot the scatter plot with different colors for each variable
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["x"],
                y=df["y"],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df['color_variable'],  # Use the assigned color variable
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=color_variable),
                ),
                text=df['patient_id'],  # Display patient ID when hovering
                hovertemplate="<b>Patient ID:</b> %{text}<extra></extra>",
            )
        )
        fig.update_traces(marker=dict(color=df['color_variable']))  # Update marker color based on selected variable


    fig.update_layout(transition_duration=200)

   # Determine which component triggered the callback
    triggered_component = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    # Handle clicks on the data table
    if triggered_component == "data-table":
        # Convert selected_rows_data_table to a set for efficient handling
        selected_rows_set = set(selected_rows_data_table)
        # Update the scatter trace to highlight selected points
        selected_indices = [
            data[data["patient_id"].eq(derived_virtual_data[row]["patient_id"])].index[0]
            for row in selected_rows_set
        ]
        fig["data"][0]["selectedpoints"] = selected_indices
        fig["data"][0]["selected"] = dict(marker=dict(color="red", size=10))
        # Update current_selected_rows with the latest selections from the data table
        current_selected_rows = list(selected_rows_set)

    # Handle clicks on the cluster graph
    if triggered_component == "cluster-graph" and click_data and click_data.get("points"):
        clicked_point = click_data["points"][0]
        if "pointIndex" in clicked_point:
            point_index = clicked_point["pointIndex"]
            if point_index not in current_selected_rows:
                current_selected_rows.append(point_index)
            else:
                current_selected_rows.remove(point_index)

    # Now, update the scatter trace to highlight selected points
    if current_selected_rows:
        selected_indices = [
            data[data["patient_id"].eq(derived_virtual_data[row]["patient_id"])].index[0]
            for row in current_selected_rows
        ]
        fig["data"][0]["selectedpoints"] = selected_indices
        fig["data"][0]["selected"] = dict(marker=dict(color="red", size=10))

    cache[cache_key] = (fig, current_selected_rows)
    return fig, current_selected_rows
