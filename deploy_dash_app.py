import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import logging
from fairness_reweight import reweight_samples_with_community
from utils import setup_logging
from load_community_definitions import load_community_definitions

# Setup logging
setup_logging()

# Load community definitions
community_defs = load_community_definitions()

# Load data
try:
    data = pd.read_csv("data/real_hr_data.csv")
except FileNotFoundError:
    print("Error: real_hr_data.csv not found. Please add your dataset to data/real_hr_data.csv.")
    exit(1)

# Initialize app
app = dash.Dash(__name__)
server = app.server

# Layout with updated sections and styles
app.layout = html.Div(className="app-container", children=[
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
    html.Div(className="content", children=[
        html.Div(className="header", children=[
            html.H1("Racial Fairness Dashboard"),
        ]),
        html.Div(className="card-container", children=[
            html.Div(className="card", children=[
                html.H3("Outcome Distribution"),
                dcc.Graph(id='outcome-distribution-graph', className="dash-graph"),
            ]),
            html.Div(className="card boost-card", children=[
                html.H3("Fairness Metrics"),
                html.Div(id='fairness-metric-result'),
            ]),
            html.Div(className="card upload-card", children=[
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
            html.Div(className="card report-card", children=[
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
    Input('fairness-metric-toggle', 'value')
)
def update_graph(reweighting, fairness_metric):
    df = data.copy()

    if reweighting == 'reweighted':
        df = reweight_samples_with_community(
            df, race_col='race', outcome_col='hired', favorable='Yes',
            community_defs=community_defs
        )
        logging.info("Applied community-driven reweighting.")
    else:
        df['sample_weight'] = 1.0
        logging.info("Using original data weights.")

    # Compute race-outcome distributions
    race_outcomes = (
        df.groupby(['race', 'hired'])['sample_weight']
        .sum()
        .groupby(level=0)
        .apply(lambda x: x / x.sum())
        .unstack(fill_value=0)
        .reset_index()
    )

    # Make sure columns exist
    for col in ['Yes', 'No']:
        if col not in race_outcomes.columns:
            race_outcomes[col] = 0

    # Melt for plot
    race_outcomes_melted = race_outcomes.melt(id_vars='race', value_vars=['Yes', 'No'],
                                              var_name='hired', value_name='proportion')

    fig = px.bar(race_outcomes_melted, x='race', y='proportion', color='hired',
                  title='Outcome Distribution by Race',
                  labels={'proportion': 'Proportion', 'race': 'Race', 'hired': 'Outcome'},
                  barmode='stack')

    # Fairness metric
    if fairness_metric == 'DI':
        white_rate = df.loc[df['race'] == 'White', 'hired'].value_counts(normalize=True).get('Yes', 0)
        black_rate = df.loc[df['race'] == 'Black', 'hired'].value_counts(normalize=True).get('Yes', 0)
        latinx_rate = df.loc[df['race'] == 'Latinx', 'hired'].value_counts(normalize=True).get('Yes', 0)
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
