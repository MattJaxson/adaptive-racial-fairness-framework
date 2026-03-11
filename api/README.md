# Racial Fairness Bias Audit API

A FastAPI service that audits datasets for racial disparate impact and generates fairness-aware sample weights.

---

## Running the server

```bash
# From the project root
uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000`.

Interactive docs (Swagger UI) are at `http://localhost:8000/docs`.

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_KEYS` | No | `dev-key-12345` | Comma-separated list of valid API keys checked via the `X-API-Key` request header. |
| `COMMUNITY_DEFS_PATH` | No | `data/community_definitions.json` | Path to the community fairness definitions JSON file used by the reweighting service. |

Example:

```bash
export API_KEYS="prod-key-abc,ci-key-xyz"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## Authentication

All endpoints (except `/health`, `/docs`, `/redoc`, `/openapi.json`) require the `X-API-Key` header.

```
X-API-Key: dev-key-12345
```

A missing or invalid key returns `401 Unauthorized`.

---

## Endpoints

### `GET /health`

Returns service health and version.

```bash
curl http://localhost:8000/health
```

---

### `POST /audit` — JSON body

Audit a dataset provided inline as a JSON list of row dicts.

```bash
curl -s -X POST http://localhost:8000/audit \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"race": "White",  "hired": "yes"},
      {"race": "White",  "hired": "yes"},
      {"race": "White",  "hired": "yes"},
      {"race": "White",  "hired": "no"},
      {"race": "Black",  "hired": "yes"},
      {"race": "Black",  "hired": "no"},
      {"race": "Black",  "hired": "no"},
      {"race": "Latinx", "hired": "yes"},
      {"race": "Latinx", "hired": "no"},
      {"race": "Latinx", "hired": "no"}
    ],
    "race_col": "race",
    "outcome_col": "hired",
    "favorable_value": "yes",
    "privileged_group": "White"
  }' | python3 -m json.tool
```

---

### `POST /audit/csv` — CSV file upload

Audit a dataset provided as a CSV file via `multipart/form-data`.

```bash
curl -s -X POST http://localhost:8000/audit/csv \
  -H "X-API-Key: dev-key-12345" \
  -F "file=@/path/to/your/dataset.csv" \
  -F "race_col=race" \
  -F "outcome_col=hired" \
  -F "favorable_value=yes" \
  -F "privileged_group=White" | python3 -m json.tool
```

---

### `POST /reweight` — JSON body

Reweight a dataset provided inline as a JSON list of row dicts. Returns each row with an added `sample_weight` column.

```bash
curl -s -X POST http://localhost:8000/reweight \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"race": "White",  "hired": "yes"},
      {"race": "White",  "hired": "yes"},
      {"race": "White",  "hired": "yes"},
      {"race": "White",  "hired": "no"},
      {"race": "Black",  "hired": "yes"},
      {"race": "Black",  "hired": "no"},
      {"race": "Black",  "hired": "no"},
      {"race": "Latinx", "hired": "yes"},
      {"race": "Latinx", "hired": "no"},
      {"race": "Latinx", "hired": "no"}
    ],
    "race_col": "race",
    "outcome_col": "hired",
    "favorable_value": "yes"
  }' | python3 -m json.tool
```

---

### `POST /reweight/csv` — CSV file upload

Reweight a dataset provided as a CSV file via `multipart/form-data`.

```bash
curl -s -X POST http://localhost:8000/reweight/csv \
  -H "X-API-Key: dev-key-12345" \
  -F "file=@/path/to/your/dataset.csv" \
  -F "race_col=race" \
  -F "outcome_col=hired" \
  -F "favorable_value=yes" | python3 -m json.tool
```

---

## Response format — `/audit`

```json
{
  "status": "success",
  "summary": {
    "total_records": 10,
    "groups_analyzed": ["Black", "Latinx", "White"],
    "outcome_column": "hired",
    "favorable_value": "yes",
    "flagged_groups": ["Black", "Latinx"]
  },
  "metrics": {
    "disparity_score": 0.4167,
    "group_outcomes": {"Black": 0.3333, "Latinx": 0.3333, "White": 0.75},
    "disparate_impact": {"Black": 0.4444, "Latinx": 0.4444, "White": 1.0},
    "statistical_parity_gap": 41.67
  },
  "findings": [
    "Black applicants had a favorable outcome rate of 33% compared to 75% for the reference group (White), a Disparate Impact ratio of 0.44 — well below the 0.8 legal threshold.",
    "The overall Statistical Parity Gap across all groups is 42 percentage points."
  ],
  "recommendation": "Immediate review recommended. 2 group(s) fall below the 0.8 Disparate Impact threshold (Black, Latinx), which may indicate discriminatory outcomes under the 4/5ths rule."
}
```

## Response format — `/reweight`

```json
{
  "status": "success",
  "records": 10,
  "reweighted_data": [
    {"race": "White", "hired": "yes", "sample_weight": 1.0},
    ...
  ],
  "summary": {
    "original_group_rates": {"Black": 0.3333, "Latinx": 0.3333, "White": 0.75},
    "target_group": "White",
    "priority_groups": ["Black", "Latinx"]
  }
}
```
