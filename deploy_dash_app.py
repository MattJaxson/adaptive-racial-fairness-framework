from racial_bias_score import calculate_racial_bias_score
from dash import dcc, html, Input, Output
import dash
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

# Define column names
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

# Load data
# Load data
try:
    data = pd.read_csv("data/real_hr_data.csv")
    print("✅ Columns detected:", data.columns.tolist())
    data.columns = data.columns.str.strip().str.lower()
    data.rename(columns={'hired': 'income'}, inplace=True)


    # Strip whitespace in critical columns
    data['income'] = data['income'].astype(str).str.strip()
    data['race'] = data['race'].astype(str).str.strip()
    
    data['income'] = data['income'].replace({
        '>50K': 'Yes', '<=50K': 'No'
    })

    # ✅ Print sample rows and distributions for debugging
    print("\n✅ Sample of loaded data:")
    print(data.head())
    print("\n📊 Income value counts:")
    print(data['income'].value_counts())
    print("\n📊 Race value counts:")
    print(data['race'].value_counts())
except FileNotFoundError:
    print("❌ Error: real_hr_data.csv not found. Please add your dataset to data/real_hr_data.csv.")
    exit(1)


# Initialize Dash app
app = dash.Dash(__name__)
server = app.server
app.title = "Racial Fairness Dashboard"

# Layout
app.layout = html.Div(className="app-container", children=[
    html.Div(className="sidebar", children=[
        html.H1("Equity Audit"),
        html.P("Navigation / filters go here."),
        dcc.RadioItems(
            id='reweight-toggle',
            options=[
                {'label': 'Original', 'value': 'original'},
                {'label': 'Reweighted', 'value': 'reweighted'}
            ],
            value='original',
            labelStyle={'display': 'block'}
        ),
        dcc.RadioItems(
            id='fairness-metric-toggle',
            options=[
                {'label': 'Disparate Impact', 'value': 'DI'},
                {'label': 'Equalized Odds', 'value': 'EO'}
            ],
            value='DI',
            labelStyle={'display': 'block'}
        )
    ]),

    html.Div(className="content", children=[
        html.Div(className="header", children=[
            html.H1("Racial Fairness Dashboard"),
            html.Div([
                html.Button("Toggle Dark Mode", id="dark-mode-toggle", className="toggle-dark"),
                html.Button("High Contrast", id="contrast-toggle", className="toggle-dark")
            ], style={'display': 'flex', 'gap': '10px'})
        ]),

        html.Div(className="card-container", children=[
            html.Div(className="card", children=[
                html.H3("Outcome Distribution"),
                dcc.Graph(id='outcome-distribution-graph', className="dash-graph"),
            ]),

            html.Div(className="card boost-card", children=[
                html.H3("Fairness Metrics"),
                html.Div(id='fairness-metric-result', children=[]),
            ]),

            html.Div(className="card boost-card", children=[
                html.H3("Racial Bias Score"),
                html.Div(id='racial-bias-score-result', children=[]),
            ]),
        ])
    ]),

    # ✅ JS loaded last for safety
    html.Script(src="/assets/animations.js")
])

# Callbacks
@app.callback(
    Output('outcome-distribution-graph', 'figure'),
    Output('fairness-metric-result', 'children'),
    Output('racial-bias-score-result', 'children'),
    Input('reweight-toggle', 'value'),
    Input('fairness-metric-toggle', 'value')
)
def update_graph(reweighting, fairness_metric):
    df = data.copy()

    # Reweighting logic
    if reweighting == 'reweighted':
        df = reweight_samples_with_community(
          df, race_col='race', outcome_col='income', favorable='Yes',
          community_defs=community_defs
        )
        logging.info("Applied community-driven reweighting.")
    
    else:
        df['sample_weight'] = 1.0
        logging.info("Using original data weights.")
    # ✅ Ensure numeric weights (fix for agg error)
    df['sample_weight'] = pd.to_numeric(df['sample_weight'], errors='coerce').fillna(1.0)
    
    # 🔍 Optional: Debug sample
    print("\n🔍 Sample of processed sample_weight column:")
    print(df[['race', 'income', 'sample_weight']].head())
    print(df['sample_weight'].dtype)

    # Group and normalize
    grouped = df.groupby(['race', 'income'], as_index=False)['sample_weight'].sum()
    grouped['sample_weight'] = grouped['sample_weight'].astype(float)
    grouped['proportion'] = grouped.groupby('race')['sample_weight'].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0
    )
    race_outcomes = grouped.pivot(index='race', columns='income', values='proportion').fillna(0).reset_index()

    for col in ['No', 'Yes']:
        if col not in race_outcomes.columns:
            race_outcomes[col] = 0

    race_outcomes_melted = race_outcomes.melt(
        id_vars='race',
        value_vars=['No', 'Yes'],
        var_name='income',
        value_name='proportion'
    )
    
    print("\n✅ Melted Data for Plot:")
    print(race_outcomes_melted.head())


    fig = px.bar(
        race_outcomes_melted, x='race', y='proportion', color='income',
        title='Outcome Distribution by Race',
        labels={'proportion': 'Proportion', 'race': 'Race', 'income': 'Income'},
        barmode='stack'
    )

    # Fairness metric
    if fairness_metric == 'DI':
        white_rate = df.loc[df['race'] == 'White', 'income'].value_counts(normalize=True).get('Yes', 0)
        black_rate = df.loc[df['race'] == 'Black', 'income'].value_counts(normalize=True).get('Yes', 0)
        latinx_rate = df.loc[df['race'] == 'Latinx', 'income'].value_counts(normalize=True).get('Yes', 0)

        di_black_white = black_rate / white_rate if white_rate else 0
        di_latinx_white = latinx_rate / white_rate if white_rate else 0
        result_text = f"Disparate Impact - Black vs White: {di_black_white:.2f} | Latinx vs White: {di_latinx_white:.2f}"
    else:
        result_text = "Equalized Odds not yet implemented."

    # Bias score
    bias_results = calculate_racial_bias_score(df, sensitive_column='race', outcome_column='income')
    group_outcomes = bias_results["group_outcomes"]
    disparity_score = bias_results["racial_disparity_score"]

    return (
        fig,
        html.P(result_text),
        html.Div(children=[
            html.P(f"Disparity Score: {disparity_score}"),
            html.Ul([html.Li(f"{group}: {score}") for group, score in group_outcomes.items()])
        ], style={'marginTop': '10px'})
    )

# Run app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8050)
