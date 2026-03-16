<p align="center">
  <h1 align="center">Adaptive Racial Fairness Framework</h1>
  <p align="center">
    <strong>The first fairness auditing tool where communities — not researchers — define what "fair" means.</strong>
  </p>
  <p align="center">
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
    <a href="#"><img src="https://img.shields.io/badge/Tests-57%20passing-brightgreen.svg" alt="Tests: 57 passing"></a>
    <a href="#"><img src="https://img.shields.io/badge/CDF-v1.0-orange.svg" alt="CDF v1.0"></a>
    <a href="https://adaptive-racial-fairness-framework.onrender.com/health"><img src="https://img.shields.io/badge/API-Live-success.svg" alt="API: Live"></a>
  </p>
</p>

---

## The Problem

Every fairness tool in production today — AIF360, Fairlearn, What-If Tool — hardcodes the EEOC's 4/5ths rule (Disparate Impact ≥ 0.80) as the fairness threshold. Researchers pick the number. Communities live with the consequences.

**What we found:** Black mortgage applicants in Michigan have a Disparate Impact ratio of 0.8174. They *technically pass* the federal 0.80 threshold — by 0.0174 points. A community asked to set their own threshold chose 0.88. Under that standard, the same group fails.

The gap between 0.80 and 0.88 is where policy quietly fails people.

## The Insight

**Fairness thresholds are governance outputs, not researcher inputs.**

This framework treats the question "what counts as fair?" as a structured community decision with full provenance — who set the threshold, when, through what process — not a default buried in a config file.

## What This Does

| Capability | Description |
|---|---|
| **Interactive Dashboard** | Upload any CSV, map columns, see disparate impact and statistical parity in real time |
| **Community-Defined Fairness (CDF) v1.0** | Open standard for encoding community fairness decisions as portable, signed JSON configs |
| **REST API** | 9 endpoints — audit, reweight, remediate, debias, generate PDF reports, check compliance |
| **Multi-Domain Validation** | Tested across hiring (HR), lending (HMDA/CFPB), and criminal justice (COMPAS/ProPublica) |
| **Regulatory Mapping** | Outputs map directly to Michigan HB 4668 and NYC Local Law 144 reporting requirements |
| **Toolkit Integration** | Adapters for AIF360 and Fairlearn that inject community governance into existing pipelines |

## Quick Start

```bash
git clone https://github.com/MattJaxson/adaptive-racial-fairness-framework.git
cd adaptive-racial-fairness-framework
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Launch the dashboard:**
```bash
python deploy_dash_app.py
```
Open [http://localhost:8050](http://localhost:8050). Click **"Try Demo — HMDA Lending Data"** to see the Michigan mortgage finding in 30 seconds, or upload your own CSV.

**Launch the API:**
```bash
uvicorn api.main:app --reload
```
API docs at [http://localhost:8000/docs](http://localhost:8000/docs).

## Demo: The Michigan Finding

The dashboard includes a one-click demo using real HMDA mortgage data from Michigan (4,463 records, CFPB):

```
Group                          DI Ratio    EEOC (0.80)    Community (0.85)
─────────────────────────────────────────────────────────────────────────
White                          1.00        PASS           PASS
Black or African American      0.8174      PASS ← here    FAIL ← here
Asian                          0.9312      PASS           PASS
```

The same group passes or fails depending on who sets the threshold. That's the whole point.

## CDF v1.0 — The Standard

Community-Defined Fairness configs are JSON documents that encode three decisions:

```json
{
  "config_version": "1.0",
  "jurisdiction": "Michigan",
  "domain": "mortgage_lending",
  "priority_groups": ["Black or African American", "Hispanic or Latino"],
  "fairness_threshold": 0.88,
  "fairness_target": "White",
  "provenance": {
    "session_id": "uuid-here",
    "date": "2026-03-15",
    "method": "community_session",
    "participant_count": 12
  }
}
```

CDF configs are tool-agnostic. Any fairness toolkit can consume them. The formal spec is at [`specs/CDF_SPECIFICATION_v1.md`](specs/CDF_SPECIFICATION_v1.md) with a JSON Schema at [`specs/community_fairness_config_v1.schema.json`](specs/community_fairness_config_v1.schema.json).

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/audit` | JSON payload audit |
| `POST` | `/audit/csv` | CSV upload audit |
| `POST` | `/audit/pdf` | CSV upload → PDF report download |
| `POST` | `/audit/remediate` | Full loop: audit → reweight → compare DI before/after |
| `POST` | `/audit/debias` | Adversarial debiasing via ExponentiatedGradient |
| `POST` | `/audit/compliance` | Validate against any CDF v1.0 community config |
| `POST` | `/reweight` | JSON payload reweight |
| `POST` | `/reweight/csv` | CSV upload reweight |

Live API: `https://adaptive-racial-fairness-framework.onrender.com`

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Layer 1: Governance                    │
│  Community sessions → CDF v1.0 configs → Config registry │
│  (WHO sets the threshold, WHEN, through WHAT process)    │
├─────────────────────────────────────────────────────────┤
│                    Layer 2: Execution                     │
│  Audit engine │ Reweighting │ Debiasing │ Compliance     │
│  Uses thresholds FROM Layer 1, never hardcoded defaults  │
├─────────────────────────────────────────────────────────┤
│                    Layer 3: Delivery                      │
│  Dash dashboard │ FastAPI │ PDF reports │ Adapters       │
└─────────────────────────────────────────────────────────┘
```

The key architectural decision: **no audit executes without a traceable config.** The governance layer is not optional — it's the precondition.

## Project Structure

```
├── deploy_dash_app.py              # Dash dashboard (main UI)
├── api/main.py                     # FastAPI service (9 endpoints)
├── fairness_audit.py               # Disparate impact & group outcome utilities
├── fairness_reweight.py            # Community-driven sample reweighting
├── racial_bias_score.py            # Disparity scoring engine
├── adversarial_fairlearn.py        # ML debiasing via Fairlearn
├── community_input.py              # Community config builder with provenance
├── report_generator.py             # PDF report generation
├── load_community_definitions.py   # Config loader with fallback defaults
├── integrations/
│   ├── aif360_adapter.py           # AIF360 + community governance
│   ├── fairlearn_adapter.py        # Fairlearn + community governance
│   └── compliance_adapter.py       # Regulatory report mapping
├── specs/
│   ├── CDF_SPECIFICATION_v1.md     # Formal CDF v1.0 standard
│   └── community_fairness_config_v1.schema.json
├── registry/                       # Community config registry
│   └── michigan/lending/           # Organized by jurisdiction/domain
├── data/external/
│   ├── hmda_michigan_lending.csv   # 4,463 HMDA mortgage records (CFPB)
│   └── compas_recidivism.csv       # 7,214 COMPAS records (ProPublica)
├── tests/
│   └── test_fairness_pipeline.py   # 57 adversarial test cases
├── docs/                           # Research documentation
│   ├── preprint_draft.md           # arXiv paper skeleton
│   ├── patent_claims.md            # Method & System patent claims
│   └── community_input_protocol.md # 90-min facilitation guide
└── validation_study.py             # Three-way validation runner
```

## Testing

```bash
# Run full adversarial test suite (57 tests)
python -m pytest tests/test_fairness_pipeline.py -v

# Quick smoke test
python test_racial_score.py

# Reproducibility check (22 verification points)
python reproduce.py
```

The test suite covers adversarial edge cases: missing reference groups, zero favorable outcomes, malformed headers, single-group datasets, threshold boundaries, and more.

## Research

This framework accompanies a research paper (in preparation) demonstrating that:

1. **Threshold sensitivity is non-trivial.** Groups within 0.02 points of the EEOC threshold flip between PASS and FAIL under modest community adjustments.
2. **Three-domain validation.** The finding holds across hiring, lending, and criminal justice datasets.
3. **The bootstrapped proxy methodology** uses published survey data on algorithmic fairness preferences to derive statistically defensible community configs when direct session data is pending.

Key datasets: HMDA/CFPB (Michigan mortgage lending), ProPublica COMPAS (Broward County recidivism).

## Citation

```bibtex
@software{jackson2026adaptive,
  author = {Jackson, Matt},
  title = {Adaptive Racial Fairness Framework: Community-Defined Fairness for Algorithmic Auditing},
  year = {2026},
  url = {https://github.com/MattJaxson/adaptive-racial-fairness-framework}
}
```

## License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  <em>Fairness thresholds should be set by the people they affect.</em>
</p>
