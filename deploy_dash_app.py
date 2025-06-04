import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import logging
from fairness_reweight import reweight_samples_with_community
from utils import setup_logging
from load_community_definitions import load_community_definitions

# Set up logging
setup_logging()

# Load community definitions
community_defs = load_community_definitions()

# Load data (using CSV as default example)
try:
    data = pd.read_csv("data/real_hr_data.csv")
except FileNotFoundError:
    print("Error: real_hr_data.csv not found. Please add your dataset to data/real_hr_data.csv.")
    exit(1)

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # For production deployment

app.layout = html.Div([
    html.H1("Racial Fairness Analysis in HR Data"),
    html.P("Toggle reweighting and fairness metric to see impact."),

    html.Div([
        html.Label("Reweighting:"),
        dcc.RadioItems(
            id='reweight-toggle',
            options=[
                {'label': 'Original', 'value': 'original'},
                {'label': 'Reweighted (Community)', 'value': 'reweighted'}
            ],
            value='original',
            inline=True
        ),
    ]),

    html.Div([
        html.Label("Fairness Metric:"),
        dcc.RadioItems(
            id='fairness-metric-toggle',
            options=[
                {'label': 'Disparate Impact', 'value': 'DI'},
                {'label': 'Equalized Odds', 'value': 'EO'}
            ],
            value='DI',
            inline=True
        ),
    ]),

    dcc.Graph(id='outcome-distribution-graph'),
    html.Div(id='fairness-metric-result')
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

    logging.info("Processed race_outcomes successfully.")
    print("Columns in race_outcomes:", race_outcomes.columns)
    print("race_outcomes:\n", race_outcomes)

    # Melt to long-form for plotting
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

    logging.info(result_text)
    return fig, html.P(result_text)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8050)
