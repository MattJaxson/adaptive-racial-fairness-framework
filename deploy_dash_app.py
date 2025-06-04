from dash import dcc, html, Input, Output
import dash
import plotly.express as px
import pandas as pd
import logging
from fairness_reweight import reweight_samples_with_community
from utils import setup_logging
from load_community_definitions import load_community_definitions
import base64
import io

# Setup logging
setup_logging()

# Load community definitions
community_defs = load_community_definitions()

# Define column names based on your dataset
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

# Load data (using CSV as default example)
try:
    data = pd.read_csv("data/real_hr_data.csv", names=column_names, header=None)  # Use names if there's no header
except FileNotFoundError:
    print("Error: real_hr_data.csv not found. Please add your dataset to data/real_hr_data.csv.")
    exit(1)

# Initialize app
app = dash.Dash(__name__)
server = app.server

# Layout with custom classes that match your styles
app.layout = html.Div(className="app-container", children=[
    # Link the JS file from the assets folder
    html.Script(src="/assets/animations.js"),  # This will automatically load the JS

    # Sidebar section
    html.Div(className="sidebar", children=[
        html.H1("Equity Audit", className="sidebar-title"),
        html.P("by Matt Jaxson", className="sidebar-email"),
        html.P("Level up your HR insights", className="sidebar-tagline"),
        html.Div(className="sidebar-nav", children=[
            html.Label("Reweighting:"),
            dcc.RadioItems(
                id='reweight-toggle',
                options=[
                    {'label': 'Original', 'value': 'original'},
                    {'label': 'Reweighted (Community)', 'value': 'reweighted'}
                ],
                value='original',
                className="sidebar-link"
            ),
            html.Label("Fairness Metric:"),
            dcc.RadioItems(
                id='fairness-metric-toggle',
                options=[
                    {'label': 'Disparate Impact', 'value': 'DI'},
                    {'label': 'Equalized Odds', 'value': 'EO'}
                ],
                value='DI',
                className="sidebar-link"
            )
        ]),
    ]),

    # Main content section
    html.Div(className="content", children=[
        html.Div(className="header", children=[
            html.H1("Racial Fairness Dashboard"),
        ]),

        # Card containers for layout
        html.Div(className="card-container", children=[
            html.Div(className="card", children=[
                html.H3("Outcome Distribution"),
                dcc.Graph(id='outcome-distribution-graph', className="dash-graph"),
            ]),

            html.Div(className="card boost-card", children=[
                html.H3("Fairness Metrics"),
                html.Div(id='fairness-metric-result'),
            ]),

            # HR Data upload section
            html.Div(className="card", children=[
                html.H3("Upload HR Data"),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px 0',
                        'backgroundColor': '#fff',
                        'color': '#a00000'
                    },
                    multiple=False
                ),
                html.Div(id='upload-status')
            ]),

            # Fairness Reports Section
            html.Div(className="card", children=[
                html.H3("Fairness Reports"),
                html.P("View or download detailed fairness reports.", style={'margin': '10px 0'}),
                html.Button("Download Report", className="button"),
                html.Button("View Full Report", className="button", style={'marginLeft': '10px'})
            ])
        ])
    ])
])

@app.callback(
    Output('outcome-distribution-graph', 'figure'),
    Output('fairness-metric-result', 'children'),
    Input('reweight-toggle', 'value'),
    Input('fairness-metric-toggle', 'value'),
    Input('upload-data', 'contents')  # Added upload-data as input
)
def update_graph(reweighting, fairness_metric, file_contents):
    if file_contents is None:
        return dash.no_update  # Exit early if no file uploaded

    # Decode and process the uploaded file
    content_type, content_string = file_contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        # Read the CSV data
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), header=None)  # No header in .data files by default
        df.columns = column_names  # Set column names manually after reading the file

        # Display the first few rows of the data to confirm it's loading properly
        print("Data loaded:\n", df.head())  # This will print the first 5 rows in the console
        print("Columns in data:", df.columns)  # Print the column names for debugging

    except Exception as e:
        print(f"Error reading file: {e}")
        return dash.no_update  # Return nothing if there's an error

    # Ensure 'race' column exists
    if 'race' not in df.columns:
        print("Error: 'race' column not found in the data")
        return dash.no_update

    # Apply reweighting logic (same as before)
    if reweighting == 'reweighted':
        df = reweight_samples_with_community(
            df, race_col='race', outcome_col='income', favorable='>50K',  # Changed from 'hired' to 'income'
            community_defs=community_defs
        )
        logging.info("Applied community-driven reweighting.")
    else:
        df['sample_weight'] = 1.0
        logging.info("Using original data weights.")

    # Continue with your race-outcome calculations and plot generation here...
    race_outcomes = (
        df.groupby(['race', 'income'])['sample_weight']  # Use 'income' instead of 'hired'
        .sum()
        .groupby(level=0)
        .apply(lambda x: x / x.sum())
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ['<=50K', '>50K']:  # Adjusted based on 'income' column
        if col not in race_outcomes.columns:
            race_outcomes[col] = 0

    race_outcomes_melted = race_outcomes.melt(id_vars='race', value_vars=['<=50K', '>50K'],
                                              var_name='income', value_name='proportion')

    fig = px.bar(race_outcomes_melted, x='race', y='proportion', color='income',
                  title='Outcome Distribution by Race',
                  labels={'proportion': 'Proportion', 'race': 'Race', 'income': 'Income'},
                  barmode='stack')

    # Fairness metric (Disparate Impact)
    if fairness_metric == 'DI':
        white_rate = df.loc[df['race'] == 'White', 'income'].value_counts(normalize=True).get('>50K', 0)
        black_rate = df.loc[df['race'] == 'Black', 'income'].value_counts(normalize=True).get('>50K', 0)
        latinx_rate = df.loc[df['race'] == 'Latinx', 'income'].value_counts(normalize=True).get('>50K', 0)
        di_black_white = black_rate / white_rate if white_rate else 0
        di_latinx_white = latinx_rate / white_rate if white_rate else 0
        result_text = f"Disparate Impact - Black vs White: {di_black_white:.2f} | Latinx vs White: {di_latinx_white:.2f}"
    else:
        result_text = "Equalized Odds not yet implemented."

    return fig, html.P(result_text)

@app.callback(
    Output('upload-status', 'children'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def update_output(contents):
    if contents is not None:
        return html.P("File uploaded successfully!", style={'color': '#a00000'})
    else:
        return html.P("No file uploaded.", style={'color': 'gray'})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8050)
