import dash
from dash import html, dcc, callback, Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import dash_html_components as html


dash.register_page(__name__, suppress_callback_exceptions=True)
data = pd.read_csv("METABRIC_RNA_Mutation2.csv")
available_variables = list(data.columns)


layout = html.Div(children=[
    # First row: Patient choosing dropdown and Overall surviving status text
    html.Div(className="row", children=[
        html.Div(className="six columns", children=[
            html.H1(children=' '),
            html.Label("Select Patient ID:"),
            dcc.Dropdown(
                id='patient-id-dropdown',
                options=[{'label': str(patient_id), 'value': patient_id} for patient_id in data['patient_id'].unique()],
                value=data['patient_id'].iloc[13],  # Set default value
            ),
        ]),
    ]),

    # Second row: Patient information table, spider chart, and mutation count pie chart
    html.Div(className="row", children=[
        html.Div(className="six columns", style={'padding': '10px', 'border': '1px solid #ccc', 'border-radius': '5px', 'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.1)', 'margin-right': '10px'}, children=[
            html.H2(children='Patient Information:'),
            html.Table(id='patient-info-table',
                       children=[
                           html.Tr([html.Th("Variable Name"), html.Th("Value")])
                       ]),
        ]),
        html.Div(className="six columns", style={'padding': '10px', 'border': '1px solid #ccc', 'border-radius': '5px', 'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.1)', 'margin-left': '10px'}, children=[
            dcc.Graph(
                id='spider-chart',
                figure={'layout': {'width': 350, 'height': 285}},  
            ),
            dcc.Graph(
                id='mutation-pie-chart',
                figure={'layout': {'width': 350, 'height': 285}}, 
            ),
        ]),
    ]),

    # Third row: Heatmap and barchart
    html.Div(className="row", children=[
        html.Div(className="six columns", style={'padding': '10px', 'border': '1px solid #ccc', 'border-radius': '5px', 'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.1)', 'margin-right': '10px'}, children=[
            dcc.Graph(id='mrna-heatmap'),
        ]),
        html.Div(className="six columns", style={'padding': '10px', 'border': '1px solid #ccc', 'border-radius': '5px', 'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.1)', 'margin-left': '10px'}, children=[
            dcc.Graph(id='gene-zscore-bar-chart'),
        ]),
    ]),
])


# Function to generate Spider Chart
def generate_spider_chart(patient_id):
    # Filter data for the selected patient ID
    selected_patient_data = data[data['patient_id'] == patient_id]

    # Select the variables for the spider chart
    spider_variables = ['neoplasm_histologic_grade',
                        'overall_survival_months', 'integrative_cluster',
                        'nottingham_prognostic_index']

    # Create the spider chart traces
    spider_chart_trace = go.Scatterpolar(
        r=selected_patient_data[spider_variables].values.tolist()[0],
        theta=spider_variables,
        fill='toself',
        name='Patient ' + str(patient_id)
    )

    return {
        'data': [spider_chart_trace],
        'layout': go.Layout(
            title=f'Spider Chart for Patient {patient_id}',
            polar=dict(
                radialaxis=dict(
                    visible=True,
                ),
            ),
        )
    }

def generate_patient_info_table(patient_id):
    # Filter data for the selected patient ID
    selected_patient_data = data[data['patient_id'] == patient_id]

    # Select the variables for the patient information table
    patient_info_variables = ['age_at_diagnosis', 'cancer_type_detailed',
                              'type_of_breast_surgery', 'chemotherapy',
                              'hormone_therapy', 'inferred_menopausal_state',
                              'lymph_nodes_examined_positive', 'tumor_size',
                              'tumor_stage']

    # Create the patient information table as a list of HTML table rows
    patient_info_rows = []
    for col in patient_info_variables:
        variable_name = col.replace("_", " ").title()
        variable_value = selected_patient_data[col].values[0]

        # If the column is Chemotherapy or Hormone Therapy, replace the value
        # 1 with "Yes" and 0 with "No"
        if col == 'chemotherapy' or col == 'hormone_therapy':
            variable_value = "Yes" if variable_value == 1 else "No"

        patient_info_rows.append(html.Tr([html.Td(variable_name), html.Td(variable_value)]))

    # Add Overall Survival Status row
    overall_survival_status = selected_patient_data['death_from_cancer'].values[0]
    patient_info_rows.append(html.Tr([html.Td("Overall Survival Status"), html.Td(overall_survival_status)]))

    return patient_info_rows





# Function to generate the pie chart of mutation_count
def generate_mutation_pie_chart(patient_id):
    # Filter data for the selected patient ID
    selected_patient_data = data[data['patient_id'] == patient_id]

    # Get mutation_count value
    mutation_count = selected_patient_data['mutation_count'].values[0]

    # If mutation_count is null or 0 or empty, display "-" in the pie chart
    if pd.isnull(mutation_count) or mutation_count == 0 or mutation_count == "":
        mutation_count = '-'
        chart_title = f'No data available about mutation count for Patient {patient_id}'
    else:
        chart_title = f'Mutation Count Pie Chart for Patient {patient_id}'

    # Create the pie chart trace
    pie_chart_trace = go.Pie(
        values=[mutation_count],
        labels=['Mutation Count'],
        hole=0.6,
        textinfo='label+value',  # Show label (Mutation Count) and value
        hoverinfo='label+value',  # Show label (Mutation Count) and value on hover
        text=[str(mutation_count)],
        showlegend=False, 
    )

    return {
        'data': [pie_chart_trace],
        'layout': go.Layout(
            title=chart_title,
        )
    }

# Function to generate the heatmap of mRNA levels for 331 genes
def generate_mrna_heatmap(patient_id):
    # Filter data for the selected patient ID
    selected_patient_data = data[data['patient_id'] == patient_id]

    # Select the mRNA levels for the 331 genes
    mrna_data = selected_patient_data.iloc[:, 31:362].values

    # Create the heatmap trace
    heatmap_trace = go.Heatmap(
        z=mrna_data,
        x=data.columns[31:362],  # Gene names
        y=['mRNA Levels'],
        colorscale='Viridis'
    )

    return {
        'data': [heatmap_trace],
        'layout': go.Layout(
            title=f'mRNA Levels Heatmap for Patient {patient_id}',
            xaxis=dict(title='Genes'),
            yaxis=dict(title=''),
        )
    }

# Updated function to generate the gene z-score bar chart
def generate_gene_zscore_bar_chart(patient_id):
    # Filter data for the selected patient ID
    selected_patient_data = data[data['patient_id'] == patient_id]

    # Select the gene z-score values for columns 520 to the end
    gene_mut_zscore_data = selected_patient_data.iloc[:, 520:]

    columns_with_non_zero_values = []

    # Loop through the columns and check for non-zero values
    for col in gene_mut_zscore_data.columns:
        if selected_patient_data[col].values[0] != 0 and selected_patient_data[col].values[0] != '0':
            columns_with_non_zero_values.append(selected_patient_data[col].name.replace('_mut', ''))

    # Select the gene z-score values for columns 31 to 520
    gene_zscore_data = selected_patient_data.iloc[:, 31:326].values.tolist()[0]

    # Create the bar chart trace
    bar_chart_trace = go.Bar(
        x=data.columns[31:326],  # Gene names
        y=gene_zscore_data,
        marker=dict(
            color=['red' if gene_name in columns_with_non_zero_values else 'blue' for gene_name in data.columns[31:326]],
        ),
    )


    # Create the figure for the bar chart
    bar_chart_figure = {
        'data': [bar_chart_trace],
        'layout': go.Layout(
            title=f'Gene Z-Score Mutation Bar Chart for Patient {patient_id}',
            xaxis=dict(title='Genes'),
            yaxis=dict(title='Z-Score'),
        )
    }

    return bar_chart_figure

# Callback to update all elements when the patient ID is changed
@callback(
    [Output('patient-info-table', 'children'),
     Output('mrna-heatmap', 'figure'),
     Output('spider-chart', 'figure'),
     Output('mutation-pie-chart', 'figure'),
     Output('gene-zscore-bar-chart', 'figure')],
    [Input('patient-id-dropdown', 'value')]
)
def update_all_elements(patient_id):
    # Generate patient information table
    patient_info_table_rows = generate_patient_info_table(patient_id)

    # Generate heatmap of mRNA levels
    mrna_heatmap_figure = generate_mrna_heatmap(patient_id)

    # Generate spider chart
    spider_chart_figure = generate_spider_chart(patient_id)

    # Generate pie chart of mutation_count
    mutation_pie_chart_figure = generate_mutation_pie_chart(patient_id)


    # Generate gene z-score bar chart
    gene_zscore_bar_chart_figure = generate_gene_zscore_bar_chart(patient_id)

    return patient_info_table_rows, mrna_heatmap_figure, spider_chart_figure, mutation_pie_chart_figure, gene_zscore_bar_chart_figure
