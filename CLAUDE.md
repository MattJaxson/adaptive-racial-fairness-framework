# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run the Dash dashboard:**
```bash
python deploy_dash_app.py
```
Dashboard runs at `http://localhost:8050`.

**Run tests:**
```bash
python test_racial_score.py          # Quick bias score smoke test
python -m pytest tests/              # Run test suite
```

**Install dependencies:**
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Architecture

The project is a Dash web application for auditing racial fairness in HR/hiring datasets. The main entrypoint is `deploy_dash_app.py`.

**Data flow:**
1. User uploads any CSV via the in-browser upload component — no hardcoded file path required.
2. Three dropdowns appear for column mapping: sensitive/race column, outcome column, and favorable outcome value (populated dynamically from the uploaded file's columns and unique values).
3. `load_community_definitions.py` reads `data/community_definitions.json` at startup for reweighting config. Falls back to defaults (`priority_groups: ["Black", "Latinx"]`, `fairness_target: "White"`) if missing. When a file is uploaded, priority groups fall back to all unique values in the selected race column if not set in the JSON.
4. The main callback either passes data through raw or calls `fairness_reweight.reweight_samples_with_community()` to assign `sample_weight` per row.
5. Metrics computed: Disparate Impact ratio (each group vs. the highest-rate group, with ⚠ flag below 0.8) or Statistical Parity gap (max − min hire rate), plus a Disparity Score from `racial_bias_score.calculate_racial_bias_score()`.

**Callback chain in `deploy_dash_app.py`:**
- `store_upload` — parses upload → stores DataFrame as JSON in `dcc.Store`, populates column dropdowns
- `update_favorable_options` — triggered by outcome column selection → fills favorable-value dropdown
- `update_dashboard` — triggered by any control change → renders all three panels

**Core modules:**
- `deploy_dash_app.py` — Dash app, layout, and three callbacks (upload, favorable-value, dashboard)
- `fairness_reweight.py` — `reweight_samples_with_community()`: per-sample weights aligning priority groups' rates to the target group
- `racial_bias_score.py` — `calculate_racial_bias_score()`: per-group mean outcomes + disparity score (max − min)
- `fairness_audit.py` — `group_outcomes_by_race()` and `disparate_impact()`: standalone utilities, not wired into the dashboard
- `adversarial_fairlearn.py` — `adversarial_fairness_pipeline()`: ML fairness mitigation via `fairlearn` ExponentiatedGradient; standalone, not wired into the dashboard
- `data_loader.py` — `load_data()`: CSV/SQL loader with cleaning; not used by the dashboard (upload is handled in-callback)
- `load_community_definitions.py` — Reads `data/community_definitions.json`
- `utils.py` — `setup_logging()` configures `basicConfig` logging

**Data requirements:**
- No fixed file path required — users upload CSV through the browser UI.
- `data/community_definitions.json` is optional; defaults are used if absent.
- Uploaded CSV must have at least one column usable as a sensitive attribute and one as a binary/categorical outcome.

**Assets:**
- `assets/styles.css` and `assets/animations.js` are auto-served by Dash from the `assets/` directory.
