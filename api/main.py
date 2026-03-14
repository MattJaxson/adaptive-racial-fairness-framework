"""
FastAPI application — Racial Fairness Bias Audit Service.
"""

from __future__ import annotations

import io
import logging
import sys
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

# ---------------------------------------------------------------------------
# Bootstrap: make the project root importable so we can import core modules.
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from racial_bias_score import calculate_racial_bias_score  # noqa: E402
from fairness_reweight import reweight_samples_with_community  # noqa: E402
from fairness_audit import disparate_impact  # noqa: E402
from load_community_definitions import load_community_definitions  # noqa: E402
from community_input import validate_community_config, is_community_valid  # noqa: E402
from report_generator import generate_pdf_report  # noqa: E402
from adversarial_fairlearn import adversarial_fairness_pipeline  # noqa: E402

from api.auth import APIKeyMiddleware  # noqa: E402
from api.models import JSONAuditRequest, JSONReweightRequest  # noqa: E402

DI_THRESHOLD_DEFAULT = 0.8  # EEOC 4/5ths rule — used only when community config has no threshold
MAX_UPLOAD_MB = 50
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Community definitions — loaded once at startup.
# ---------------------------------------------------------------------------
_COMMUNITY_DEFS_PATH = os.environ.get(
    "COMMUNITY_DEFS_PATH",
    str(Path(PROJECT_ROOT) / "data" / "community_definitions.json"),
)

community_defs: dict = {}


def _load_community_defs() -> dict:
    defs = load_community_definitions(_COMMUNITY_DEFS_PATH)
    logger.info(
        "Community definitions loaded — target: %s, priority groups: %s",
        defs.get("fairness_target"),
        defs.get("priority_groups"),
    )
    return defs


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Racial Fairness Bias Audit API",
    description="Audit datasets for racial disparate impact and generate fairness-aware sample weights.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(APIKeyMiddleware)


@app.on_event("startup")
async def startup_event() -> None:
    global community_defs
    community_defs = _load_community_defs()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _coerce_favorable(df: pd.DataFrame, outcome_col: str, favorable_value: str) -> tuple[pd.DataFrame, Any]:
    """
    Try to coerce the favorable_value to match the dtype of outcome_col.
    Returns the (possibly mutated) dataframe and the coerced favorable value.
    """
    col_dtype = df[outcome_col].dtype
    try:
        if pd.api.types.is_integer_dtype(col_dtype):
            return df, int(favorable_value)
        if pd.api.types.is_float_dtype(col_dtype):
            return df, float(favorable_value)
        if pd.api.types.is_bool_dtype(col_dtype):
            return df, favorable_value.lower() in ("1", "true", "yes")
    except (ValueError, TypeError):
        pass
    # Default: keep as string; also stringify the outcome column for safe comparison.
    df[outcome_col] = df[outcome_col].astype(str)
    return df, favorable_value


def _validate_columns(df: pd.DataFrame, race_col: str, outcome_col: str) -> None:
    missing = [c for c in (race_col, outcome_col) if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Column(s) not found in data: {missing}. Available columns: {list(df.columns)}",
        )
    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty.")


def _compute_group_rates(df: pd.DataFrame, race_col: str, outcome_col: str, favorable: Any) -> dict[str, float]:
    """Return favorable outcome rate per group as a plain dict (vectorized)."""
    binary = (df[outcome_col] == favorable).astype(float)
    rates = binary.groupby(df[race_col]).mean().round(4)
    return {str(k): float(v) for k, v in rates.items()}


def _build_audit_report(
    df: pd.DataFrame,
    race_col: str,
    outcome_col: str,
    favorable_value: str,
    privileged_group: str | None,
) -> dict:
    """Core logic for /audit — shared between CSV and JSON paths."""
    df, favorable = _coerce_favorable(df, outcome_col, favorable_value)
    _validate_columns(df, race_col, outcome_col)

    # Community-defined threshold — the core differentiator of this framework.
    # If the community set a threshold, use it. Otherwise fall back to EEOC 0.8.
    di_threshold: float = float(community_defs.get("fairness_threshold", DI_THRESHOLD_DEFAULT))

    # Bias score — requires a numeric binary column
    df['_binary_outcome'] = (df[outcome_col] == favorable).astype(float)
    bias_result = calculate_racial_bias_score(df, sensitive_column=race_col, outcome_column='_binary_outcome')
    group_outcomes: dict[str, float] = {str(k): round(float(v), 4) for k, v in bias_result["group_outcomes"].items()}
    disparity_score: float = float(bias_result["racial_disparity_score"])

    # Determine reference (privileged) group.
    # Default: White. Rationale — this tool measures systemic racial disadvantage,
    # which is historically directional. White is the standard reference in EEOC
    # Disparate Impact analysis. Falls back to highest-rate group if White is not
    # present in the dataset, or uses caller-supplied privileged_group if provided.
    DEFAULT_REF_GROUP = "White"
    if privileged_group and privileged_group in group_outcomes:
        ref_group = privileged_group
    elif DEFAULT_REF_GROUP in group_outcomes:
        ref_group = DEFAULT_REF_GROUP
    else:
        if privileged_group:
            logger.warning(
                "privileged_group '%s' not found in data — falling back to highest-rate group.",
                privileged_group,
            )
        else:
            logger.info(
                "'%s' not found in dataset — falling back to highest-rate group as reference.",
                DEFAULT_REF_GROUP,
            )
        ref_group = max(group_outcomes, key=lambda g: group_outcomes[g])

    ref_rate = group_outcomes[ref_group]

    # Disparate Impact per group
    di_ratios: dict[str, float | None] = {}
    for group in group_outcomes:
        if group == ref_group:
            di_ratios[group] = 1.0
            continue
        di = disparate_impact(
            data=df,
            race_col=race_col,
            outcome_col=outcome_col,
            privileged=ref_group,
            unprivileged=group,
            favorable=favorable,
        )
        di_ratios[group] = round(float(di), 4) if di is not None else None

    # Statistical parity gap = max rate − min rate (in percentage points)
    all_rates = list(group_outcomes.values())
    stat_parity_gap = round((max(all_rates) - min(all_rates)) * 100, 2) if all_rates else 0.0

    # Flagged groups — uses community-defined threshold, not hardcoded 0.8
    flagged_groups = [g for g, di in di_ratios.items() if di is not None and di < di_threshold]

    # Plain-English findings
    findings: list[str] = []
    for group, rate in group_outcomes.items():
        if group == ref_group:
            continue
        di = di_ratios.get(group)
        pct = round(rate * 100)
        ref_pct = round(ref_rate * 100)
        if di is None:
            findings.append(
                f"{group} applicants had a favorable outcome rate of {pct}%; "
                f"Disparate Impact is undefined because the reference group ({ref_group}) "
                f"has no positive outcomes."
            )
        elif di < di_threshold:
            severity = "substantially below" if di < 0.5 else "below"
            findings.append(
                f"{group} applicants had a favorable outcome rate of {pct}% compared to "
                f"{ref_pct}% for the reference group ({ref_group}), "
                f"a Disparate Impact ratio of {di:.2f} — {severity} the {di_threshold} threshold."
            )
        else:
            findings.append(
                f"{group} applicants had a favorable outcome rate of {pct}% compared to "
                f"{ref_pct}% for the reference group ({ref_group}), "
                f"a Disparate Impact ratio of {di:.2f} — within the acceptable range."
            )

    findings.append(
        f"The overall Statistical Parity Gap across all groups is "
        f"{stat_parity_gap:.0f} percentage points."
    )

    # Recommendation
    n = len(flagged_groups)
    if flagged_groups:
        group_word = "group falls" if n == 1 else "groups fall"
        recommendation = (
            f"Immediate review recommended. {n} {group_word} below the "
            f"Disparate Impact threshold of {di_threshold} ({', '.join(flagged_groups)}), which may indicate "
            f"discriminatory outcomes under the 4/5ths rule."
        )
    else:
        recommendation = (
            "The data shows no statistically significant disparate impact across analyzed groups. "
            f"All groups meet or exceed the {di_threshold} Disparate Impact threshold."
        )

    # Determine audit type based on community config provenance
    audit_type = "community_valid" if is_community_valid(community_defs) else "standard"
    provenance = community_defs.get("provenance", {})

    return {
        "status": "success",
        "audit_type": audit_type,
        "community_config": {
            "priority_groups": community_defs.get("priority_groups", []),
            "fairness_target": community_defs.get("fairness_target", ref_group),
            "fairness_threshold": di_threshold,
            "provenance": provenance if provenance else None,
        },
        "summary": {
            "total_records": len(df),
            "groups_analyzed": list(group_outcomes.keys()),
            "outcome_column": outcome_col,
            "favorable_value": favorable_value,
            "flagged_groups": flagged_groups,
        },
        "metrics": {
            "disparity_score": disparity_score,
            "group_outcomes": group_outcomes,
            "disparate_impact": di_ratios,
            "statistical_parity_gap": stat_parity_gap,
        },
        "findings": findings,
        "recommendation": recommendation,
    }


def _build_reweight_report(
    df: pd.DataFrame,
    race_col: str,
    outcome_col: str,
    favorable_value: str,
) -> dict:
    """Core logic for /reweight — shared between CSV and JSON paths."""
    df, favorable = _coerce_favorable(df, outcome_col, favorable_value)
    _validate_columns(df, race_col, outcome_col)

    original_rates = _compute_group_rates(df, race_col, outcome_col, favorable)

    reweighted_df = reweight_samples_with_community(
        data=df.copy(),
        race_col=race_col,
        outcome_col=outcome_col,
        favorable=favorable,
        community_defs=community_defs,
    )

    reweighted_df['sample_weight'] = reweighted_df['sample_weight'].round(4)
    reweighted_records = reweighted_df.to_dict(orient="records")

    return {
        "status": "success",
        "records": len(reweighted_df),
        "reweighted_data": reweighted_records,
        "summary": {
            "original_group_rates": original_rates,
            "target_group": community_defs.get("fairness_target", "unknown"),
            "priority_groups": community_defs.get("priority_groups", []),
        },
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Health"])
async def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}


# ---------- /audit ----------------------------------------------------------

@app.post("/audit", tags=["Audit"])
async def audit_json(request: JSONAuditRequest) -> JSONResponse:
    """Audit a dataset supplied as a JSON body."""
    logger.info(
        "POST /audit (JSON) — %d records, race_col=%s, outcome_col=%s",
        len(request.data),
        request.race_col,
        request.outcome_col,
    )
    try:
        df = pd.DataFrame(request.data)
        report = _build_audit_report(
            df=df,
            race_col=request.race_col,
            outcome_col=request.outcome_col,
            favorable_value=request.favorable_value,
            privileged_group=request.privileged_group,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during /audit (JSON)")
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}") from exc

    return JSONResponse(content=report)


@app.post("/audit/csv", tags=["Audit"])
async def audit_csv(
    file: UploadFile = File(..., description="CSV file to audit."),
    race_col: str = Form(...),
    outcome_col: str = Form(...),
    favorable_value: str = Form(...),
    privileged_group: str | None = Form(default=None),
) -> JSONResponse:
    """Audit a dataset supplied as a CSV file upload (multipart/form-data)."""
    logger.info(
        "POST /audit/csv — file=%s, race_col=%s, outcome_col=%s",
        file.filename,
        race_col,
        outcome_col,
    )
    try:
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB}MB limit.")
        df = pd.read_csv(io.BytesIO(contents))
        df.columns = df.columns.str.strip()
        report = _build_audit_report(
            df=df,
            race_col=race_col,
            outcome_col=outcome_col,
            favorable_value=favorable_value,
            privileged_group=privileged_group,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during /audit/csv")
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}") from exc

    return JSONResponse(content=report)


# ---------- /audit/pdf ------------------------------------------------------

@app.post("/audit/pdf", tags=["Audit"])
async def audit_pdf(
    file: UploadFile = File(..., description="CSV file to audit."),
    race_col: str = Form(...),
    outcome_col: str = Form(...),
    favorable_value: str = Form(...),
    privileged_group: str | None = Form(default=None),
) -> Response:
    """Audit a CSV dataset and return a professionally formatted PDF report."""
    logger.info(
        "POST /audit/pdf — file=%s, race_col=%s, outcome_col=%s",
        file.filename,
        race_col,
        outcome_col,
    )
    try:
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB}MB limit.")
        df = pd.read_csv(io.BytesIO(contents))
        df.columns = df.columns.str.strip()
        report = _build_audit_report(
            df=df,
            race_col=race_col,
            outcome_col=outcome_col,
            favorable_value=favorable_value,
            privileged_group=privileged_group,
        )
        pdf_bytes = generate_pdf_report(report)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during /audit/pdf")
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}") from exc

    filename = f"fairness_audit_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------- /reweight -------------------------------------------------------

@app.post("/audit/remediate", tags=["Audit"])
async def audit_remediate(
    file: UploadFile = File(..., description="CSV file to audit and remediate."),
    race_col: str = Form(...),
    outcome_col: str = Form(...),
    favorable_value: str = Form(...),
    privileged_group: str | None = Form(default=None),
) -> JSONResponse:
    """
    Full remediation loop: audit → reweight → post-mitigation audit.
    Returns pre- and post-mitigation metrics so the delta is visible.
    """
    logger.info(
        "POST /audit/remediate — file=%s, race_col=%s, outcome_col=%s",
        file.filename,
        race_col,
        outcome_col,
    )
    try:
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB}MB limit.")
        df = pd.read_csv(io.BytesIO(contents))
        df.columns = df.columns.str.strip()

        di_threshold = float(community_defs.get("fairness_threshold", DI_THRESHOLD_DEFAULT))

        # Step 1: Pre-mitigation audit
        pre_report = _build_audit_report(
            df=df,
            race_col=race_col,
            outcome_col=outcome_col,
            favorable_value=favorable_value,
            privileged_group=privileged_group,
        )

        flagged_groups = pre_report["summary"]["flagged_groups"]
        if not flagged_groups:
            return JSONResponse(content={
                "status": "success",
                "message": "No flagged groups — remediation not required.",
                "pre_mitigation": pre_report,
                "post_mitigation": None,
                "delta": None,
            })

        # Step 2: Reweight
        df_copy, favorable = _coerce_favorable(df.copy(), outcome_col, favorable_value)
        reweighted_df = reweight_samples_with_community(
            data=df_copy,
            race_col=race_col,
            outcome_col=outcome_col,
            favorable=favorable,
            community_defs=community_defs,
        )

        # Step 3: Post-mitigation metrics — computed from weighted outcome rates.
        # Instead of simulating random outcomes (which is non-deterministic and
        # mathematically incoherent), we compute what the group outcome rates
        # *would be* under the reweighted distribution and report those directly.
        binary = (reweighted_df[outcome_col] == favorable).astype(float)
        sw = reweighted_df["sample_weight"].fillna(1.0)

        post_group_rates: dict[str, float] = {}
        for group in reweighted_df[race_col].unique():
            mask = reweighted_df[race_col] == group
            weighted_sum = (binary[mask] * sw[mask]).sum()
            weight_total = sw[mask].sum()
            post_group_rates[str(group)] = round(float(weighted_sum / weight_total), 4) if weight_total > 0 else 0.0

        # Compute post-mitigation DI using same reference group as pre-report
        pre_ref_group = privileged_group or community_defs.get("fairness_target", "White")
        if pre_ref_group not in post_group_rates:
            pre_ref_group = max(post_group_rates, key=lambda g: post_group_rates[g])
        post_ref_rate = post_group_rates.get(pre_ref_group, 0.0)

        post_di: dict[str, float | None] = {}
        for group, rate in post_group_rates.items():
            if group == pre_ref_group:
                post_di[group] = 1.0
            elif post_ref_rate == 0:
                post_di[group] = None
            else:
                post_di[group] = round(rate / post_ref_rate, 4)

        post_flagged = [g for g, di in post_di.items() if di is not None and di < di_threshold]

        post_report = {
            "status": "success",
            "audit_type": pre_report["audit_type"],
            "community_config": pre_report["community_config"],
            "summary": {
                "total_records": len(reweighted_df),
                "groups_analyzed": list(post_group_rates.keys()),
                "outcome_column": outcome_col,
                "favorable_value": favorable_value,
                "flagged_groups": post_flagged,
            },
            "metrics": {
                "disparity_score": round(max(post_group_rates.values()) - min(post_group_rates.values()), 4),
                "group_outcomes": post_group_rates,
                "disparate_impact": post_di,
                "statistical_parity_gap": round((max(post_group_rates.values()) - min(post_group_rates.values())) * 100, 2),
            },
            "note": "Post-mitigation metrics are computed from weighted outcome rates, not from a new model.",
        }

        # Step 4: Compute delta
        pre_di = pre_report["metrics"]["disparate_impact"]
        post_di = post_report["metrics"]["disparate_impact"]
        delta = {
            group: {
                "pre": pre_di.get(group),
                "post": post_di.get(group),
                "change": (
                    round(post_di[group] - pre_di[group], 4)
                    if pre_di.get(group) is not None and post_di.get(group) is not None
                    else None
                ),
            }
            for group in set(list(pre_di.keys()) + list(post_di.keys()))
        }

        return JSONResponse(content={
            "status": "success",
            "flagged_groups_pre": flagged_groups,
            "flagged_groups_post": post_report["summary"]["flagged_groups"],
            "pre_mitigation": pre_report,
            "post_mitigation": post_report,
            "delta": delta,
        })

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during /audit/remediate")
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}") from exc


@app.post("/audit/debias", tags=["Audit"])
async def audit_debias(
    file: UploadFile = File(..., description="CSV file to debias."),
    race_col: str = Form(..., description="Sensitive attribute (race/ethnicity) column name."),
    outcome_col: str = Form(..., description="Outcome column name."),
    favorable_value: str = Form(..., description="Value in outcome column that counts as favorable."),
    feature_cols: str = Form(..., description="Comma-separated list of feature columns to use for model training."),
    constraint: str = Form(default="demographic_parity", description="Fairness constraint: 'demographic_parity'."),
) -> JSONResponse:
    """
    Run adversarial debiasing via ExponentiatedGradient (fairlearn).

    Trains a baseline logistic regression and a fairness-constrained model,
    then returns pre/post mitigation accuracy, disparate impact per group,
    and a plain-English interpretation of the improvement.

    - **feature_cols**: comma-separated column names to use as model features.
      Do NOT include the sensitive attribute column — it is used only as the
      fairness constraint, not as a feature.
    - **constraint**: only `demographic_parity` is supported in v1.
    """
    logger.info(
        "POST /audit/debias — file=%s, race_col=%s, outcome_col=%s, features=%s",
        file.filename, race_col, outcome_col, feature_cols,
    )
    try:
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB}MB limit.")

        df = pd.read_csv(io.BytesIO(contents))
        df.columns = df.columns.str.strip()

        parsed_features = [c.strip() for c in feature_cols.split(",") if c.strip()]
        if not parsed_features:
            raise HTTPException(status_code=400, detail="feature_cols cannot be empty.")

        df, favorable = _coerce_favorable(df, outcome_col, favorable_value)

        result = adversarial_fairness_pipeline(
            data=df,
            feature_cols=parsed_features,
            outcome_col=outcome_col,
            sensitive_col=race_col,
            favorable_value=favorable,
            constraint=constraint,
        )
    except HTTPException:
        raise
    except (ValueError, ImportError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during /audit/debias")
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}") from exc

    return JSONResponse(content=result)


@app.post("/reweight", tags=["Reweight"])
async def reweight_json(request: JSONReweightRequest) -> JSONResponse:
    """Reweight a dataset supplied as a JSON body."""
    logger.info(
        "POST /reweight (JSON) — %d records, race_col=%s, outcome_col=%s",
        len(request.data),
        request.race_col,
        request.outcome_col,
    )
    try:
        df = pd.DataFrame(request.data)
        report = _build_reweight_report(
            df=df,
            race_col=request.race_col,
            outcome_col=request.outcome_col,
            favorable_value=request.favorable_value,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during /reweight (JSON)")
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}") from exc

    return JSONResponse(content=report)


@app.post("/reweight/csv", tags=["Reweight"])
async def reweight_csv(
    file: UploadFile = File(..., description="CSV file to reweight."),
    race_col: str = Form(...),
    outcome_col: str = Form(...),
    favorable_value: str = Form(...),
) -> JSONResponse:
    """Reweight a dataset supplied as a CSV file upload (multipart/form-data)."""
    logger.info(
        "POST /reweight/csv — file=%s, race_col=%s, outcome_col=%s",
        file.filename,
        race_col,
        outcome_col,
    )
    try:
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB}MB limit.")
        df = pd.read_csv(io.BytesIO(contents))
        df.columns = df.columns.str.strip()
        report = _build_reweight_report(
            df=df,
            race_col=race_col,
            outcome_col=outcome_col,
            favorable_value=favorable_value,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during /reweight/csv")
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}") from exc

    return JSONResponse(content=report)
