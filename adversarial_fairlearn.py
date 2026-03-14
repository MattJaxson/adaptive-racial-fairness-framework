"""
Adversarial Fairness Pipeline
-------------------------------
Wraps fairlearn's ExponentiatedGradient to produce a debiased classifier
and compare pre/post mitigation performance + fairness metrics.

Used by the /audit/debias API endpoint.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def adversarial_fairness_pipeline(
    data: pd.DataFrame,
    feature_cols: list[str],
    outcome_col: str,
    sensitive_col: str,
    favorable_value: Any,
    constraint: str = "demographic_parity",
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict:
    """
    Run adversarial fairness mitigation via ExponentiatedGradient.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset including features, outcome, and sensitive attribute.
    feature_cols : list[str]
        Columns to use as model features. Categorical columns are one-hot encoded.
        The sensitive attribute column must NOT be in this list.
    outcome_col : str
        The outcome column name.
    sensitive_col : str
        The sensitive attribute (race/ethnicity) column name.
    favorable_value : Any
        The value in outcome_col that counts as a favorable outcome (mapped to 1).
    constraint : str
        Fairness constraint to apply. Currently supports: "demographic_parity".
    test_size : float
        Fraction of data to hold out for evaluation (default: 0.3).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Pre/post mitigation classification reports and fairness metrics.

    Raises
    ------
    ImportError
        If fairlearn is not installed.
    ValueError
        If inputs are invalid.
    """
    try:
        from fairlearn.reductions import ExponentiatedGradient, DemographicParity
    except ImportError as exc:
        raise ImportError(
            "fairlearn is required for adversarial debiasing. "
            "It is included in requirements.txt."
        ) from exc

    # --- Validate inputs -------------------------------------------------------
    missing = [c for c in feature_cols + [outcome_col, sensitive_col] if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in dataset: {missing}")
    if sensitive_col in feature_cols:
        raise ValueError("sensitive_col must not be in feature_cols — remove it from features.")
    if len(data) < 50:
        raise ValueError("Dataset too small for adversarial debiasing (minimum 50 rows).")

    # --- Prepare features -------------------------------------------------------
    X_raw = data[feature_cols].copy()

    # One-hot encode any non-numeric columns
    cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X_raw = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)

    X_raw = X_raw.fillna(X_raw.median(numeric_only=True))
    feature_names = X_raw.columns.tolist()

    # --- Encode outcome ---------------------------------------------------------
    y = (data[outcome_col] == favorable_value).astype(int)

    # --- Encode sensitive attribute ---------------------------------------------
    s_raw = data[sensitive_col].astype(str)
    le = LabelEncoder()
    s = pd.Series(le.fit_transform(s_raw), index=data.index, name=sensitive_col)
    group_labels = dict(enumerate(le.classes_))

    # --- Train/test split -------------------------------------------------------
    X_train, X_test, y_train, y_test, s_train, s_test, s_raw_train, s_raw_test = train_test_split(
        X_raw, y, s, s_raw, test_size=test_size, random_state=random_state, stratify=y
    )

    # --- Baseline (no mitigation) -----------------------------------------------
    baseline = LogisticRegression(solver="liblinear", random_state=random_state, max_iter=500)
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)

    baseline_report = classification_report(y_test, y_pred_baseline, output_dict=True, zero_division=0)
    baseline_group_rates = _group_positive_rates(y_pred_baseline, s_raw_test)
    baseline_di = _disparate_impact_from_rates(baseline_group_rates)

    # --- Mitigated model --------------------------------------------------------
    if constraint == "demographic_parity":
        fairness_constraint = DemographicParity()
    else:
        raise ValueError(f"Unsupported constraint: {constraint}. Use 'demographic_parity'.")

    estimator = LogisticRegression(solver="liblinear", random_state=random_state, max_iter=500)
    mitigator = ExponentiatedGradient(estimator, constraints=fairness_constraint)
    mitigator.fit(X_train, y_train, sensitive_features=s_train)

    y_pred_mitigated = mitigator.predict(X_test)

    mitigated_report = classification_report(y_test, y_pred_mitigated, output_dict=True, zero_division=0)
    mitigated_group_rates = _group_positive_rates(y_pred_mitigated, s_raw_test)
    mitigated_di = _disparate_impact_from_rates(mitigated_group_rates)

    # --- Delta ------------------------------------------------------------------
    delta_accuracy = (
        mitigated_report.get("accuracy", 0) - baseline_report.get("accuracy", 0)
    )

    return {
        "status": "success",
        "constraint": constraint,
        "dataset_summary": {
            "total_records": len(data),
            "train_records": len(X_train),
            "test_records": len(X_test),
            "feature_cols": feature_names,
            "sensitive_col": sensitive_col,
            "outcome_col": outcome_col,
            "favorable_value": str(favorable_value),
        },
        "baseline": {
            "accuracy": round(baseline_report.get("accuracy", 0), 4),
            "classification_report": _round_report(baseline_report),
            "group_positive_rates": baseline_group_rates,
            "disparate_impact": baseline_di,
        },
        "mitigated": {
            "accuracy": round(mitigated_report.get("accuracy", 0), 4),
            "classification_report": _round_report(mitigated_report),
            "group_positive_rates": mitigated_group_rates,
            "disparate_impact": mitigated_di,
        },
        "delta": {
            "accuracy_change": round(delta_accuracy, 4),
            "fairness_improvement": {
                group: round(
                    mitigated_di.get(group, 0) - baseline_di.get(group, 0), 4
                )
                for group in set(list(baseline_di.keys()) + list(mitigated_di.keys()))
            },
        },
        "interpretation": _interpret(baseline_di, mitigated_di, delta_accuracy),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_positive_rates(y_pred: np.ndarray, s: pd.Series) -> dict[str, float]:
    """Fraction of positive predictions per sensitive group."""
    df = pd.DataFrame({"pred": y_pred, "group": s.values})
    rates = df.groupby("group")["pred"].mean().round(4)
    return {str(k): float(v) for k, v in rates.items()}


def _disparate_impact_from_rates(rates: dict[str, float]) -> dict[str, float]:
    """DI ratio for each group relative to the highest-rate group.

    Equivalent to unprivileged_rate / privileged_rate when the privileged
    (reference) group is defined as the group with the highest positive
    prediction rate — consistent with fairness_audit.disparate_impact().
    """
    if not rates:
        return {}
    max_rate = max(rates.values())
    if max_rate == 0:
        return {g: 0.0 for g in rates}
    return {g: round(r / max_rate, 4) for g, r in rates.items()}


def _round_report(report: dict) -> dict:
    """Round all numeric values in a classification_report dict."""
    rounded = {}
    for k, v in report.items():
        if isinstance(v, dict):
            rounded[k] = {mk: round(mv, 4) if isinstance(mv, float) else mv for mk, mv in v.items()}
        elif isinstance(v, float):
            rounded[k] = round(v, 4)
        else:
            rounded[k] = v
    return rounded


def _interpret(baseline_di: dict, mitigated_di: dict, delta_accuracy: float) -> str:
    """Plain-English interpretation of the mitigation result."""
    flagged_before = [g for g, v in baseline_di.items() if v < 0.8]
    flagged_after = [g for g, v in mitigated_di.items() if v < 0.8]
    resolved = [g for g in flagged_before if g not in flagged_after]
    remaining = [g for g in flagged_after]

    parts = []
    if resolved:
        parts.append(
            f"Mitigation resolved disparate impact for {len(resolved)} group(s): {', '.join(resolved)}."
        )
    if remaining:
        parts.append(
            f"{len(remaining)} group(s) still below the 0.8 DI threshold after mitigation: "
            f"{', '.join(remaining)}."
        )
    if not flagged_before:
        parts.append("No groups were below the DI threshold before mitigation.")

    acc_note = (
        f"Model accuracy changed by {delta_accuracy:+.2%} after applying the fairness constraint."
    )
    parts.append(acc_note)

    if delta_accuracy < -0.05:
        parts.append(
            "Note: accuracy decreased by more than 5 percentage points. "
            "Consider reviewing the feature set or increasing training data."
        )

    return " ".join(parts)
