import base64
import copy
import io
import logging
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.express as px
import pandas as pd

from fairness_reweight import reweight_samples_with_community
from utils import setup_logging
from load_community_definitions import load_community_definitions
from racial_bias_score import calculate_racial_bias_score

setup_logging()
community_defs = load_community_definitions()

DI_THRESHOLD = 0.8  # 4/5ths rule legal threshold

# ── Demo datasets ─────────────────────────────────────────────────────────────
DEMO_DATASETS = {
    'hmda': {
        'path': Path(__file__).parent / 'data' / 'external' / 'hmda_michigan_lending.csv',
        'label': 'HMDA Michigan Lending (4,463 records)',
        'race_col': 'derived_race',
        'outcome_col': 'action_taken',
        'favorable_value': '1',
        'description': 'Real mortgage data from CFPB. Shows Black applicants at DI=0.8174 — '
                        'passes EEOC 0.80 threshold but fails community-defined 0.85.',
    },
    'compas': {
        'path': Path(__file__).parent / 'data' / 'external' / 'compas_recidivism.csv',
        'label': 'COMPAS Recidivism (7,214 records)',
        'race_col': 'race',
        'outcome_col': 'two_year_recid',
        'favorable_value': '0',
        'description': 'ProPublica COMPAS data. African-American defendants at DI=0.8009 — '
                        'barely passes the federal standard.',
    },
}

app = dash.Dash(__name__)
server = app.server
app.title = "Equity Audit Dashboard"

# ── Layout ────────────────────────────────────────────────────────────────────

app.layout = html.Div(className="app-container", children=[

    # Sidebar
    html.Div(className="sidebar", children=[
        html.H2("Equity Audit", className="sidebar-title"),
        html.P("Racial Fairness Analysis", className="sidebar-tagline"),

        html.Hr(className="sidebar-divider"),

        # Demo buttons
        html.P("Try a Demo", className="sidebar-label"),
        html.Div([
            html.Button("HMDA Lending Data", id="demo-hmda", className="demo-btn"),
            html.Button("COMPAS Recidivism", id="demo-compas", className="demo-btn"),
        ], className="demo-btn-group"),
        html.Div(id='demo-description', className="demo-description"),

        html.Hr(className="sidebar-divider"),

        # Upload
        html.P("Or Upload Your Own", className="sidebar-label"),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag & Drop or ', html.A('Select CSV')]),
            multiple=False,
        ),
        html.Div(id='upload-status'),

        html.Hr(className="sidebar-divider"),

        # Column mapping
        html.P("Sensitive / Race Column", className="sidebar-label"),
        dcc.Dropdown(id='race-col-dropdown', placeholder='Select column…', className="sidebar-dropdown"),

        html.P("Outcome Column", className="sidebar-label"),
        dcc.Dropdown(id='outcome-col-dropdown', placeholder='Select column…', className="sidebar-dropdown"),

        html.P("Favorable Outcome Value", className="sidebar-label"),
        dcc.Dropdown(id='favorable-value-dropdown', placeholder='Select value…', className="sidebar-dropdown"),

        html.Hr(className="sidebar-divider"),

        # Controls
        html.P("Data View", className="sidebar-label"),
        dcc.RadioItems(
            id='reweight-toggle',
            options=[
                {'label': 'Original', 'value': 'original'},
                {'label': 'Reweighted', 'value': 'reweighted'},
            ],
            value='original',
            className="sidebar-radio",
        ),

        html.Br(),

        html.P("Fairness Metric", className="sidebar-label"),
        dcc.RadioItems(
            id='fairness-metric-toggle',
            options=[
                {'label': 'Disparate Impact', 'value': 'DI'},
                {'label': 'Statistical Parity', 'value': 'SP'},
            ],
            value='DI',
            className="sidebar-radio",
        ),

        html.Hr(className="sidebar-divider"),

        html.Div([
            html.Button("Dark Mode", id="dark-mode-toggle", className="toggle-dark"),
            html.Button("High Contrast", id="contrast-toggle", className="toggle-dark"),
        ], style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap'}),
    ]),

    # Main content
    html.Div(className="content", children=[
        html.Div(className="header", children=[
            html.H1("Racial Fairness Dashboard"),
        ]),
        html.Div(id='dashboard-content', children=[
            html.Div(
                "Upload a CSV and map your columns using the sidebar to get started.",
                className="empty-state",
            )
        ]),
    ]),

    dcc.Store(id='stored-data'),
])

# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output('stored-data', 'data'),
    Output('race-col-dropdown', 'options'),
    Output('race-col-dropdown', 'value'),
    Output('outcome-col-dropdown', 'options'),
    Output('outcome-col-dropdown', 'value'),
    Output('upload-status', 'children'),
    Output('demo-description', 'children'),
    Input('upload-data', 'contents'),
    Input('demo-hmda', 'n_clicks'),
    Input('demo-compas', 'n_clicks'),
    State('upload-data', 'filename'),
    prevent_initial_call=True,
)
def store_upload(contents, demo_hmda_clicks, demo_compas_clicks, filename):
    """Parse uploaded CSV or load demo dataset, store as JSON, populate column dropdowns."""
    ctx = callback_context
    if not ctx.triggered:
        return None, [], None, [], None, '', ''

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # ── Demo dataset loading ──────────────────────────────────────────────────
    if trigger_id in ('demo-hmda', 'demo-compas'):
        demo_key = 'hmda' if trigger_id == 'demo-hmda' else 'compas'
        demo = DEMO_DATASETS[demo_key]
        try:
            df = pd.read_csv(demo['path'])
            df.columns = df.columns.str.strip()
            # Convert outcome column to string for consistent matching
            df[demo['outcome_col']] = df[demo['outcome_col']].astype(str)
            col_options = [{'label': c, 'value': c} for c in df.columns]
            status = html.Span(
                f"✓ Demo: {demo['label']}",
                className="upload-success",
            )
            description = html.P(demo['description'], className="demo-desc-text")
            logging.info("Loaded demo dataset: %s — %d rows", demo_key, len(df))
            return (
                df.to_json(date_format='iso', orient='split'),
                col_options, demo['race_col'],
                col_options, demo['outcome_col'],
                status, description,
            )
        except Exception as exc:
            logging.error("Demo load failed: %s", exc)
            return None, [], None, [], None, html.Span(f"Error: {exc}", className="upload-error"), ''

    # ── User upload ───────────────────────────────────────────────────────────
    if contents is None:
        return None, [], None, [], None, '', ''

    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        df.columns = df.columns.str.strip()
        col_options = [{'label': c, 'value': c} for c in df.columns]
        status = html.Span(f"✓ {filename} ({len(df):,} rows)", className="upload-success")
        logging.info("Uploaded %s — %d rows, %d columns", filename, len(df), len(df.columns))
        return df.to_json(date_format='iso', orient='split'), col_options, None, col_options, None, status, ''
    except Exception as exc:
        logging.error("Upload failed: %s", exc)
        return None, [], None, [], None, html.Span(f"Error: {exc}", className="upload-error"), ''


@app.callback(
    Output('favorable-value-dropdown', 'options'),
    Output('favorable-value-dropdown', 'value'),
    Input('outcome-col-dropdown', 'value'),
    State('stored-data', 'data'),
    prevent_initial_call=True,
)
def update_favorable_options(outcome_col, stored_data):
    """Populate favorable-value dropdown from unique values in the outcome column."""
    if not outcome_col or not stored_data:
        return [], None
    df = pd.read_json(stored_data, orient='split')
    unique_vals = sorted(df[outcome_col].dropna().unique().tolist(), key=str)
    options = [{'label': str(v), 'value': v} for v in unique_vals]

    # Auto-select favorable value if this matches a demo dataset
    auto_value = None
    for demo in DEMO_DATASETS.values():
        if outcome_col == demo['outcome_col']:
            fav = demo['favorable_value']
            # Match against string or numeric versions
            for v in unique_vals:
                if str(v) == fav:
                    auto_value = v
                    break
            if auto_value is not None:
                break

    return options, auto_value


@app.callback(
    Output('dashboard-content', 'children'),
    Input('stored-data', 'data'),
    Input('race-col-dropdown', 'value'),
    Input('outcome-col-dropdown', 'value'),
    Input('favorable-value-dropdown', 'value'),
    Input('reweight-toggle', 'value'),
    Input('fairness-metric-toggle', 'value'),
    prevent_initial_call=True,
)
def update_dashboard(stored_data, race_col, outcome_col, favorable_value, reweighting, fairness_metric):
    """Render outcome chart, fairness metric panel, and disparity score."""
    if not stored_data or not race_col or not outcome_col or favorable_value is None:
        return html.Div(
            "Select all column mappings to view the analysis.",
            className="empty-state",
        )

    df = pd.read_json(stored_data, orient='split')
    df[race_col] = df[race_col].astype(str).str.strip()
    df[outcome_col] = df[outcome_col].astype(str).str.strip()
    favorable_str = str(favorable_value)

    # Apply reweighting
    if reweighting == 'reweighted':
        local_defs = copy.deepcopy(community_defs)
        local_defs.setdefault('priority_groups', df[race_col].unique().tolist())
        df = reweight_samples_with_community(
            df, race_col=race_col, outcome_col=outcome_col,
            favorable=favorable_str, community_defs=local_defs,
        )
        logging.info("Applied community-driven reweighting.")
    else:
        df['sample_weight'] = 1.0

    df['sample_weight'] = pd.to_numeric(df['sample_weight'], errors='coerce').fillna(1.0)

    # ── Outcome distribution chart ────────────────────────────────────────────
    grouped = df.groupby([race_col, outcome_col], as_index=False)['sample_weight'].sum()
    grouped['proportion'] = grouped.groupby(race_col)['sample_weight'].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0
    )

    fig = px.bar(
        grouped, x=race_col, y='proportion', color=outcome_col,
        title='Outcome Distribution by Group',
        labels={'proportion': 'Proportion', race_col: 'Group', outcome_col: 'Outcome'},
        barmode='stack',
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=20, l=10, r=10),
        legend_title_text='Outcome',
    )

    # ── Per-group weighted hire rates ─────────────────────────────────────────
    hire_rates = {}
    for group in df[race_col].unique():
        gdf = df[df[race_col] == group]
        favorable_weight = gdf.loc[gdf[outcome_col] == favorable_str, 'sample_weight'].sum()
        total_weight = gdf['sample_weight'].sum()
        hire_rates[group] = favorable_weight / total_weight if total_weight > 0 else 0

    # ── Fairness metric panel ─────────────────────────────────────────────────
    DEFAULT_REF_GROUP = 'White'
    if fairness_metric == 'DI':
        if DEFAULT_REF_GROUP in hire_rates:
            ref_group = DEFAULT_REF_GROUP
        else:
            ref_group = max(hire_rates, key=hire_rates.get)
        ref_rate = hire_rates[ref_group]
        rows = []
        for group, rate in sorted(hire_rates.items(), key=lambda x: -x[1]):
            di = rate / ref_rate if ref_rate > 0 else 0
            below = di < DI_THRESHOLD and group != ref_group
            rows.append(html.Li(
                [
                    html.Span(f"{group}: ", style={'fontWeight': '700'}),
                    f"{rate:.1%} → DI = {di:.2f}",
                    html.Span(" ⚠ < 0.8", className="flag-warning") if below else None,
                ],
                style={'marginBottom': '8px'},
            ))
        metric_content = html.Div([
            html.P(
                f"Reference group (highest rate): {ref_group} at {ref_rate:.1%}",
                className="metric-subtitle",
            ),
            html.P(
                "Disparate Impact = group rate ÷ reference rate. Values below 0.8 indicate potential discrimination (4/5ths rule).",
                className="metric-note",
            ),
            html.Ul(rows, style={'paddingLeft': '20px', 'marginTop': '10px'}),
        ])
    else:  # Statistical Parity
        max_rate = max(hire_rates.values())
        min_rate = min(hire_rates.values())
        gap = max_rate - min_rate
        rows = []
        for group, rate in sorted(hire_rates.items(), key=lambda x: -x[1]):
            rows.append(html.Li(
                [html.Span(f"{group}: ", style={'fontWeight': '700'}), f"{rate:.1%}"],
                style={'marginBottom': '8px'},
            ))
        metric_content = html.Div([
            html.P(f"Statistical Parity Gap: {gap:.1%}", className="metric-subtitle"),
            html.P(
                "Difference between the highest and lowest favorable outcome rates across groups. 0% = perfect parity.",
                className="metric-note",
            ),
            html.Ul(rows, style={'paddingLeft': '20px', 'marginTop': '10px'}),
        ])

    # ── Disparity score ───────────────────────────────────────────────────────
    score_df = df.copy()
    score_df['_binary'] = (score_df[outcome_col] == favorable_str).astype(float)
    bias_results = calculate_racial_bias_score(score_df, sensitive_column=race_col, outcome_column='_binary')
    disparity_score = bias_results['racial_disparity_score']

    return html.Div(className="card-container", children=[
        html.Div(className="card chart-card", children=[
            html.H3("Outcome Distribution"),
            dcc.Graph(figure=fig, className="dash-graph"),
        ]),
        html.Div(className="card boost-card", children=[
            html.H3("Fairness Metrics"),
            metric_content,
        ]),
        html.Div(className="card boost-card", children=[
            html.H3("Disparity Score"),
            html.P(f"{disparity_score:.4f}", className="score-number"),
            html.P(
                "Max − min favorable outcome rate across all groups. Score of 0 = perfect parity.",
                className="metric-note",
            ),
        ]),
    ])


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8050)
