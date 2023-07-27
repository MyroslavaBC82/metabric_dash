import dash
from dash import html, dcc, callback, Output, Input, State
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px

dash.register_page(__name__, path='/', suppress_callback_exceptions=True)
data = pd.read_csv("METABRIC_RNA_Mutation.csv")
available_variables = list(data.columns)
# Extract the numerical variables for correlation
numerical_data = data.select_dtypes(include='number')

layout = html.Div(children=[

    html.Div(className='row', children=[
        html.Div( style={'padding': '10px', 'border': '1px solid #ccc', 'border-radius': '5px', 'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.1)'}, children=[
            html.H3('Statistics for each variable'),

            # Dropdown to select the variable for statistics table
            html.Label("Choose variable to filter:"),
            dcc.Dropdown(
                id='statistics-variable-dropdown',
                options=[{'label': var, 'value': var} for var in available_variables],
                value='',  # Set default value
            ),

            # Scrollable table container for statistics table
            html.Div(
                style={'height': '400px', 'overflowY': 'scroll'},
                children=[
                    html.Table(id='statistics-table', children=[
                        html.Thead(
                            html.Tr([
                                html.Th('Variable'),
                                html.Th('Number of Values'),
                                html.Th('Number of Gaps'),
                                html.Th('Number of Unique Values'),
                                html.Th('Number of 0 Values'),
                                html.Th('Most Common Value'),
                                html.Th('Min Value'),
                                html.Th('Max Value'),
                                html.Th('Median'),
                                html.Th('Mean'),
                                html.Th('Std Value'),
                                html.Th('Number of Outliers (2 σ)'),
                                html.Th('Number of Outliers (3 σ)'),
                                html.Th('Number of Outliers (4 σ)'),
                            ])
                        ),
                        html.Tbody(id='statistics-table-body'),  # Include the statistics-table-body here
                    ]),
                ],
            ),
        ]),

    ], style={'margin-bottom': '20px'}),

    html.Div(className='row', children=[
        html.Div(className='six columns', style={'padding': '10px', 'border': '1px solid #ccc', 'border-radius': '5px', 'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.1)'}, children=[
            # html.H3('Correlation Heatmap'),
            dcc.Graph(id='correlation-heatmap'),
        ]),

        html.Div(className='six columns', style={'padding': '10px', 'border': '1px solid #ccc', 'border-radius': '5px', 'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.1)'}, children=[
            # Dropdown to select the variable for correlation table
            html.H3('Pairwise correlation'),
            html.Label("Choose variable:"),
            dcc.Dropdown(
                id='variable-dropdown',
                options=[{'label': var, 'value': var} for var in numerical_data.columns],
                value=numerical_data.columns[5],  # Set default value
            ),

            # Scrollable table container for correlation table
            html.Div(
                style={'height': '570px', 'overflowY': 'scroll'},
                children=[
                    html.Table(id='correlation-table', children=[
                        html.Thead(
                            html.Tr([
                                html.Th('Variable'),
                                html.Th('Correlation'),
                                html.Th('Diagram'),
                            ])
                        ),
                        html.Tbody(id='correlation-table-body'),  # Include the correlation-table-body here
                    ]),
                ],
            ),
        ]),
    ], style={'margin-bottom': '20px'}),

# Frequency Graph by Label section
html.Div(style={'padding': '10px', 'border': '1px solid #ccc', 'border-radius': '5px', 'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.1)', 'margin-bottom': '20px'}, children=[
    html.Div(className='row', children=[
        # Dropdown container
        html.Div( children=[
            html.Label("Choose variable for frequency graph:"),
            dcc.Dropdown(
                id='frequency-variable-dropdown',
                options=[{'label': var, 'value': var} for var in data.columns[:31]],
                value=data.columns[7],  # Set default value
            ),
        ]), 
        # Graph container
        html.Div( children=[
            html.H3('Frequency Graph by Label'),
            dcc.Graph(id='frequency-graph'),
        ]),
        
        
    ]),
]),



    # Scatter Plot and Scatter Matrix section
    html.Div(className='row', children=[
        html.Div( style={'padding': '10px', 'border': '1px solid #ccc', 'border-radius': '5px', 'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.1)'}, children=[
            html.H3('Scatter Plot'),

            # Dropdowns for X and Y axes
            html.Label("Choose variable for X-axis:"),
            dcc.Dropdown(
                id='scatter-x-variable-dropdown',
                options=[{'label': var, 'value': var} for var in numerical_data.columns],
                value=numerical_data.columns[9],  # Set default value
            ),
            html.Label("Choose variable for Y-axis:"),
            dcc.Dropdown(
                id='scatter-y-variable-dropdown',
                options=[{'label': var, 'value': var} for var in numerical_data.columns],
                value=numerical_data.columns[12],  # Set default value
            ),
            html.Div(style={'margin-top': '20px'}),

            html.Button("Clear Clicked Points", id="clear-button", n_clicks=0),
            html.Button("Show All Data", id="show-all-data-button", n_clicks=0),

            # Scatter plot container
            dcc.Graph(id='scatter-plot', selectedData=None),
        ]),

    ], style={'margin-bottom': '20px'}),

    # Scatter Matrix section 
    html.Div(className='row', children=[
        html.Div(className='twelve columns', style={'padding': '10px', 'border': '1px solid #ccc', 'border-radius': '5px', 'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.1)'}, children=[
            html.H3('Scatter Matrix'),

            # Dropdown for scatter matrix
            html.Label("Choose variables for scatter matrix:"),
            dcc.Dropdown(
                id='scatter-matrix-variables',
                options=[{'label': var, 'value': var} for var in numerical_data.columns],
                value=[numerical_data.columns[11], numerical_data.columns[12]],  # Set default value
                multi=True,  # Allow multiple variable selection
            ),
            # Scatter matrix container
            dcc.Graph(id='scatter-matrix-plot', selectedData=None),
        ]),
    ], style={'margin-bottom': '20px'}),

], className='app-container')



# Define a callback to update the correlation heatmap plot
@callback(
    dash.dependencies.Output('correlation-heatmap', 'figure'),
    [dash.dependencies.Input('statistics-variable-dropdown', 'value')]
)
def update_correlation_heatmap(selected_variable):
    # Calculate the correlation matrix between all numerical variables
    correlation_matrix = numerical_data.corr()

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',  # You can choose any colorscale you prefer
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Correlation'),
    ))

    layout = go.Layout(
        title='Correlation Heatmap',
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0),
        height=700,
    )
    fig.update_layout(layout)

    return fig

# Define a callback to update the selected variable based on click data
@callback(
    Output('variable-dropdown', 'value'),
    [Input('correlation-heatmap', 'clickData')],
    [State('variable-dropdown', 'value')]
)
def update_selected_variable_from_click_data(click_data, selected_variable):
    if click_data is not None:
        selected_variable = click_data['points'][0]['x']
    return selected_variable

# Define a callback to update the correlation table when a new variable is selected in the dropdown
@callback(
    Output('correlation-table-body', 'children'),
    [Input('variable-dropdown', 'value')]
)
def update_correlation_table(selected_variable):
    if selected_variable is not None:
        # Calculate pairwise correlations
        correlations = numerical_data.corr()[selected_variable].reset_index()
        correlations.columns = ['Variable', 'Correlation']

        # Sort correlations by absolute values in descending order
        correlations = correlations.iloc[correlations['Correlation'].abs().sort_values(ascending=False).index]

        # Generate the table rows with correlation bars
        table_rows = [
            html.Tr([
                html.Td(correlations.iloc[i]['Variable']),
                html.Td(f"{correlations.iloc[i]['Correlation']:.2f}"),
                html.Td(html.Div(style={
                    'width': f"{abs(correlations.iloc[i]['Correlation']) * 50}px",
                    'height': '20px',
                    'background-color': 'orange' if correlations.iloc[i]['Correlation'] > 0 else 'blue',
                    'margin-right': '10px',  # Add some space between the correlation value and the bar
                    'position': 'relative',  # Set position to relative for proper alignment
                })),
            ])
            for i in range(len(correlations))
        ]

        return table_rows

    # If no variable is selected, return an empty table
    return []



# Define a callback to update the statistics table
@callback(
    dash.dependencies.Output('statistics-table-body', 'children'),
    [dash.dependencies.Input('statistics-variable-dropdown', 'value')]
)
def update_statistics_table(selected_variable):
    # Calculate statistics for all variables
    statistics = []
    for column in data.columns:
        num_values = data[column].count()
        num_gaps = data[column].isnull().sum()
        num_unique_values = data[column].nunique()
        most_common_value = data[column].mode().values[0]

        if pd.api.types.is_numeric_dtype(data[column]):
            # For numerical variables, calculate additional statistics
            num_zeros = (data[column] == 0).sum()
            min_value = data[column].min()
            max_value = data[column].max()
            median = data[column].median()
            mean = round(data[column].mean(), 2)
            std_value = round(data[column].std(), 2)
            num_outliers_2sigma = data[(data[column] - mean).abs() > 2 * std_value].shape[0]
            num_outliers_3sigma = data[(data[column] - mean).abs() > 3 * std_value].shape[0]
            num_outliers_4sigma = data[(data[column] - mean).abs() > 4 * std_value].shape[0]
            num_text_values = '-'
        else:
            num_zeros = '-'
            min_value = '-'
            max_value = '-'
            median = '-'
            mean = '-'
            std_value = '-'
            num_outliers_2sigma = '-'
            num_outliers_3sigma = '-'
            num_outliers_4sigma = '-'

        statistics.append([
            column, num_values, num_gaps, num_unique_values, num_zeros,
            most_common_value, min_value, max_value, median, mean, std_value,
            num_outliers_2sigma, num_outliers_3sigma, num_outliers_4sigma,
        ])

    # Filter rows based on the selected variable
    if selected_variable:
        statistics = [row for row in statistics if row[0] == selected_variable]

    # Generate the table rows for statistics
    table_rows = [
        html.Tr([html.Td(stat) for stat in stat_row])
        for stat_row in statistics
    ]

    return table_rows


# Define a callback to update the frequency graph
@callback(
    dash.dependencies.Output('frequency-graph', 'figure'),
    [dash.dependencies.Input('frequency-variable-dropdown', 'value')]
)
def update_frequency_graph(selected_variable):
    # Calculate the frequency distribution by label
    frequency_data = data.groupby(selected_variable).size().reset_index(name='Frequency')

    # Get a list of unique labels for the selected variable
    labels = frequency_data[selected_variable]

    # Create a color scale for the bars based on the number of labels
    colors = [f"hsl({hue}, 50%, 50%)" for hue in range(0, 360, int(360 / len(labels)))]

    # Create the bar chart
    trace = go.Bar(
        x=frequency_data[selected_variable],
        y=frequency_data['Frequency'],
        marker=dict(color=colors)
    )

    layout = go.Layout(
        title=f'Frequency Distribution of {selected_variable}',
        xaxis=dict(title=selected_variable),
        yaxis=dict(title='Frequency'),
    )

    figure = go.Figure(data=[trace], layout=layout)

    return figure


def find_matching_variable(data, selected_value):
    for column in data.columns:
        if selected_value in data[column].values:
            return column
    return None

# Somewhere in your code, initialize the clicked_patient_ids set as an empty set
clicked_patient_ids = set()


# Define a callback to update the scatter plot based on the selected variables and clicked points
@callback(
    Output('scatter-plot', 'figure'),
    [Input('scatter-x-variable-dropdown', 'value'),
     Input('scatter-y-variable-dropdown', 'value'),
     Input('scatter-plot', 'clickData'),
     Input('scatter-matrix-plot', 'clickData'),
     Input('frequency-graph', 'clickData'),
     Input('clear-button', 'n_clicks'),
     Input('show-all-data-button', 'n_clicks')]  # New input for the "Show All Data" button
)
def update_scatter_plot(x_variable, y_variable, scatter_click_data, matrix_click_data, frequency_click_data, clear_button_clicks, show_all_data_clicks):
    # Access the global variable clicked_patient_ids
    global clicked_patient_ids

    # Update the clicked_patient_ids set based on the clickData from both scatter plots and frequency chart
    if scatter_click_data:
        patient_id = scatter_click_data['points'][0]['text']
        clicked_patient_ids.add(patient_id)
    if matrix_click_data:
        patient_id = matrix_click_data['points'][0]['text']
        clicked_patient_ids.add(patient_id)

    # Determine whether to show all data or filtered data
    if show_all_data_clicks is not None and show_all_data_clicks > 0:
        filtered_data = data.copy()  # Show all data (no filtering)
    else:
            # Update the clicked_patient_ids set based on the clickData from both scatter plots and frequency chart
        if frequency_click_data:
            selected_value = frequency_click_data['points'][0]['x']
            selected_variable = find_matching_variable(data, selected_value)
            if selected_variable is not None:
                filtered_data = data[data[selected_variable] == selected_value]
                clicked_patient_ids.intersection_update(set(filtered_data['patient_id']))  # Keep only IDs present in filtered data
            else:
                filtered_data = data.copy()  # If selected_variable is None, no filtering is done
        else:
            filtered_data = data.copy()  # If no frequency_click_data, no filtering is done

    if clear_button_clicks is not None and clear_button_clicks > 0:
        clicked_patient_ids = set()



    # Create the scatter plot
    trace = go.Scatter(
        x=filtered_data[x_variable],
        y=filtered_data[y_variable],
        mode='markers',
        marker=dict(
            size=8,
            color=[
                'red' if patient_id in clicked_patient_ids else 'blue'
                for patient_id in filtered_data['patient_id']
            ],
            opacity=0.7,
            line=dict(width=0)
        ),
        text=filtered_data['patient_id'],  # Set patient IDs as hover text
        hovertemplate="Patient ID: %{text}<br>"  # Define the hover template
                     f"{x_variable}: %{{x}}<br>"
                     f"{y_variable}: %{{y}}<extra></extra>",  # Include the variable values in hover
    )

    layout = go.Layout(
        title='Scatter Plot',
        xaxis=dict(title=x_variable),
        yaxis=dict(title=y_variable),
    )

    figure = go.Figure(data=[trace], layout=layout)

    return figure


# Define a callback to update the scatter matrix plot based on the selected variables and clicked points
@callback(
    Output('scatter-matrix-plot', 'figure'),
    [Input('scatter-matrix-variables', 'value'),
     Input('scatter-plot', 'clickData'),
     Input('scatter-matrix-plot', 'clickData'),
     Input('frequency-graph', 'clickData'),
     Input('clear-button', 'n_clicks'),
     Input('show-all-data-button', 'n_clicks')]  # New input for the "Show All Data" button
)
def update_scatter_matrix(selected_variables, scatter_click_data, matrix_click_data, frequency_click_data, clear_button_clicks, show_all_data_clicks):
    # Access the global variable clicked_patient_ids
    global clicked_patient_ids

    # Update the clicked_patient_ids set based on the clickData from both scatter plots and frequency chart
    if scatter_click_data:
        patient_id = scatter_click_data['points'][0]['text']
        clicked_patient_ids.add(patient_id)
    if matrix_click_data:
        patient_id = matrix_click_data['points'][0]['text']
        clicked_patient_ids.add(patient_id)

    # Determine whether to show all data or filtered data
    if show_all_data_clicks is not None and show_all_data_clicks > 0:
        filtered_data = data.copy()  # Show all data (no filtering)
    else:
        if frequency_click_data:
            selected_value = frequency_click_data['points'][0]['x']
            selected_variable = find_matching_variable(data, selected_value)
            if selected_variable is not None:
                filtered_data = data[data[selected_variable] == selected_value]
                clicked_patient_ids.intersection_update(set(filtered_data['patient_id']))  # Keep only IDs present in filtered data
            else:
                filtered_data = data.copy()  # If selected_variable is None, no filtering is done
        else:
            filtered_data = data.copy()  # If no frequency_click_data, no filtering is done

    # Create the scatter matrix plot
    num_variables = len(selected_variables)
    fig = make_subplots(rows=num_variables, cols=num_variables, shared_xaxes=True, shared_yaxes=True)

    if clear_button_clicks is not None and clear_button_clicks > 0:
        clicked_patient_ids = set()


    # Loop through each pair of selected variables and create the scatter plot or bar plot accordingly
    for i in range(num_variables):
        for j in range(num_variables):
            x_var = selected_variables[i]
            y_var = selected_variables[j]

            if i == j:
                # Create bar plot for diagonal elements
                frequency_data = filtered_data[x_var].value_counts().reset_index()
                frequency_data.columns = [x_var, 'Frequency']
                trace = go.Bar(
                    x=frequency_data[x_var],
                    y=frequency_data['Frequency'],
                    marker=dict(
                        color=[
                            'red' if patient_id in clicked_patient_ids else 'blue'
                            for patient_id in filtered_data['patient_id']
                        ],
                    ),
                    text=frequency_data[x_var],  # Set patient IDs as hover text
                    hovertemplate="Patient ID: %{text}<br>"  # Define the hover template
                                 f"{x_var}: %{{x}}<br>"
                                 f"Frequency: %{{y}}<extra></extra>",  # Include the variable values in hover
                )
                fig.add_trace(trace, row=i + 1, col=j + 1)
            else:
                # Create scatter plot for off-diagonal elements
                trace = go.Scatter(
                    x=filtered_data[x_var],
                    y=filtered_data[y_var],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=[
                            'red' if patient_id in clicked_patient_ids else 'blue'
                            for patient_id in filtered_data['patient_id']
                        ],
                        opacity=0.7,
                        line=dict(width=0)
                    ),
                    text=filtered_data['patient_id'],  # Set patient IDs as hover text
                    hovertemplate="Patient ID: %{text}<br>"  # Define the hover template
                                 f"{x_var}: %{{x}}<br>"
                                 f"{y_var}: %{{y}}<extra></extra>",  # Include the variable values in hover
                )
                fig.add_trace(trace, row=i + 1, col=j + 1)

    # Update layout of the scatter matrix plot
    layout = go.Layout(
        title='Scatter Matrix',
        showlegend=False,
        height=800,
        width=1400,
    )
    fig.update_layout(layout)

    return fig

