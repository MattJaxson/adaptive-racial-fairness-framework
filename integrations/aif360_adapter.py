"""
AIF360 Integration Adapter
===========================
Bridges community-defined fairness configurations into IBM's AIF360 toolkit.

This adapter allows any organization already using AIF360 to layer community
governance on top without switching tools. The community config defines the
parameters; AIF360 does the computation.

Usage:
    from integrations.aif360_adapter import CommunityAIF360Audit

    audit = CommunityAIF360Audit.from_config("data/community_definitions.json")
    results = audit.run(df, label_col="hired", protected_col="race")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CommunityAIF360Audit:
    """
    Wraps AIF360's BinaryLabelDataset and metrics with community-defined parameters.

    The key difference from using AIF360 directly: the privileged group, unprivileged
    groups, and flagging threshold come from a community configuration with provenance —
    not from the auditor's discretion.
    """

    def __init__(self, config: dict):
        self.config = config
        self.priority_groups = config["priority_groups"]
        self.fairness_target = config["fairness_target"]
        self.threshold = config.get("fairness_threshold", 0.8)
        self.provenance = config.get("provenance", {})
        self.audit_classification = config.get("audit_classification", "standard")

    @classmethod
    def from_config(cls, config_path: str) -> CommunityAIF360Audit:
        """Load a community config from a JSON file."""
        with open(config_path) as f:
            config = json.load(f)
        return cls(config)

    @classmethod
    def from_dict(cls, config: dict) -> CommunityAIF360Audit:
        """Create from an in-memory config dict."""
        return cls(config)

    def run(
        self,
        df: pd.DataFrame,
        label_col: str,
        protected_col: str,
        favorable_label: object = 1,
    ) -> dict:
        """
        Run a community-governed audit using AIF360.

        Falls back to a pure-pandas implementation if AIF360 is not installed,
        so the community config format works regardless of toolkit availability.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to audit.
        label_col : str
            Column containing the outcome (0/1 or categorical).
        protected_col : str
            Column containing the racial/ethnic group.
        favorable_label : object
            The value in label_col that represents a favorable outcome.

        Returns
        -------
        dict
            Audit results including DI ratios, flagged groups, and provenance.
        """
        try:
            return self._run_aif360(df, label_col, protected_col, favorable_label)
        except ImportError:
            logger.info("AIF360 not installed — using built-in DI computation.")
            return self._run_builtin(df, label_col, protected_col, favorable_label)

    def _run_aif360(
        self, df, label_col, protected_col, favorable_label
    ) -> dict:
        """Run using AIF360's BinaryLabelDatasetMetric."""
        from aif360.datasets import BinaryLabelDataset
        from aif360.metrics import BinaryLabelDatasetMetric

        # Prepare binary label
        df_work = df.copy()
        df_work["__label__"] = (df_work[label_col] == favorable_label).astype(int)

        # AIF360 requires numeric protected attributes
        group_map = {g: i for i, g in enumerate(sorted(df_work[protected_col].unique()))}
        df_work["__protected__"] = df_work[protected_col].map(group_map)

        dataset = BinaryLabelDataset(
            df=df_work[["__label__", "__protected__"]],
            label_names=["__label__"],
            protected_attribute_names=["__protected__"],
            favorable_label=1,
            unfavorable_label=0,
        )

        # Privileged group = community-defined reference
        ref_code = group_map.get(self.fairness_target)
        if ref_code is None:
            # Fall back to highest-rate group
            rates = df_work.groupby("__protected__")["__label__"].mean()
            ref_code = rates.idxmax()

        results = {"groups": {}, "provenance": self.provenance, "audit_classification": self.audit_classification}
        reverse_map = {v: k for k, v in group_map.items()}

        ref_rate = df_work[df_work["__protected__"] == ref_code]["__label__"].mean()

        for code, group_name in reverse_map.items():
            metric = BinaryLabelDatasetMetric(
                dataset,
                unprivileged_groups=[{"__protected__": code}],
                privileged_groups=[{"__protected__": ref_code}],
            )
            di = metric.disparate_impact()
            rate = df_work[df_work["__protected__"] == code]["__label__"].mean()

            results["groups"][group_name] = {
                "outcome_rate": round(rate, 4),
                "disparate_impact": round(di, 4) if di is not None else None,
                "flagged": di is not None and di < self.threshold and code != ref_code,
                "is_priority_group": group_name in self.priority_groups,
            }

        results["reference_group"] = self.fairness_target
        results["threshold"] = self.threshold
        results["flagged_groups"] = [
            g for g, d in results["groups"].items() if d["flagged"]
        ]
        return results

    def _run_builtin(
        self, df, label_col, protected_col, favorable_label
    ) -> dict:
        """Pure-pandas fallback — no AIF360 dependency required."""
        binary = (df[label_col] == favorable_label).astype(float)
        group_rates = binary.groupby(df[protected_col]).mean().to_dict()

        # Reference group
        ref = self.fairness_target
        if ref not in group_rates:
            ref = max(group_rates, key=lambda g: group_rates[g])

        ref_rate = group_rates[ref]

        results = {
            "reference_group": ref,
            "threshold": self.threshold,
            "audit_classification": self.audit_classification,
            "provenance": self.provenance,
            "groups": {},
        }

        for group, rate in group_rates.items():
            if group == ref:
                di = 1.0
            elif ref_rate == 0:
                di = None
            else:
                di = round(rate / ref_rate, 4)

            results["groups"][group] = {
                "outcome_rate": round(rate, 4),
                "disparate_impact": di,
                "flagged": di is not None and di < self.threshold and group != ref,
                "is_priority_group": group in self.priority_groups,
            }

        results["flagged_groups"] = [
            g for g, d in results["groups"].items() if d["flagged"]
        ]
        return results


def validate_config_schema(config: dict) -> tuple[bool, list[str]]:
    """
    Validate a config dict against the CDF v1.0 schema.
    Returns (is_valid, list_of_issues).
    """
    issues = []

    if not config.get("priority_groups"):
        issues.append("Missing or empty 'priority_groups'")
    if not config.get("fairness_target"):
        issues.append("Missing 'fairness_target'")

    threshold = config.get("fairness_threshold")
    if threshold is None or not (0 < threshold <= 1):
        issues.append("'fairness_threshold' must be between 0 and 1")

    provenance = config.get("provenance")
    if not provenance:
        issues.append("Missing 'provenance' — config cannot be community-valid without provenance")
    else:
        for field in ["record_id", "input_protocol", "input_date", "input_participants"]:
            if not provenance.get(field):
                issues.append(f"Provenance missing required field: '{field}'")

    return len(issues) == 0, issues
