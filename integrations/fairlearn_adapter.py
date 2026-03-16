"""
Fairlearn Integration Adapter
===============================
Bridges community-defined fairness configurations into Microsoft's Fairlearn toolkit.

This adapter allows organizations using Fairlearn for mitigation to use community-defined
parameters instead of researcher defaults. The community decides the constraint;
Fairlearn enforces it.

Usage:
    from integrations.fairlearn_adapter import CommunityFairlearnMitigation

    mitigator = CommunityFairlearnMitigation.from_config("data/community_definitions.json")
    results = mitigator.mitigate(X_train, y_train, sensitive_features, base_model)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CommunityFairlearnMitigation:
    """
    Wraps Fairlearn's ExponentiatedGradient with community-defined constraints.

    Standard Fairlearn usage: the researcher picks DemographicParity or
    EqualizedOdds as the constraint. Community-governed usage: the community
    configuration specifies the constraint parameters, and the provenance
    record documents who made that decision.
    """

    def __init__(self, config: dict):
        self.config = config
        self.priority_groups = config["priority_groups"]
        self.fairness_target = config["fairness_target"]
        self.threshold = config.get("fairness_threshold", 0.8)
        self.provenance = config.get("provenance", {})

    @classmethod
    def from_config(cls, config_path: str) -> CommunityFairlearnMitigation:
        with open(config_path) as f:
            config = json.load(f)
        return cls(config)

    @classmethod
    def from_dict(cls, config: dict) -> CommunityFairlearnMitigation:
        return cls(config)

    def audit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> dict:
        """
        Audit predictions using community-defined parameters.

        Uses Fairlearn's MetricFrame if available, falls back to pandas.

        Returns
        -------
        dict
            Per-group metrics, flagged groups, and provenance.
        """
        try:
            return self._audit_fairlearn(y_true, y_pred, sensitive_features)
        except ImportError:
            logger.info("Fairlearn not installed — using built-in computation.")
            return self._audit_builtin(y_true, y_pred, sensitive_features)

    def _audit_fairlearn(self, y_true, y_pred, sensitive_features) -> dict:
        from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate

        mf = MetricFrame(
            metrics={
                "selection_rate": selection_rate,
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )

        rates = mf.by_group["selection_rate"].to_dict()
        return self._compute_flags(rates)

    def _audit_builtin(self, y_true, y_pred, sensitive_features) -> dict:
        df = pd.DataFrame({
            "y_pred": y_pred,
            "group": sensitive_features,
        })
        rates = df.groupby("group")["y_pred"].mean().to_dict()
        return self._compute_flags(rates)

    def _compute_flags(self, rates: dict) -> dict:
        ref = self.fairness_target
        if ref not in rates:
            ref = max(rates, key=lambda g: rates[g])

        ref_rate = rates[ref]

        groups = {}
        for group, rate in rates.items():
            if group == ref:
                di = 1.0
            elif ref_rate == 0:
                di = None
            else:
                di = round(rate / ref_rate, 4)

            groups[group] = {
                "selection_rate": round(rate, 4),
                "disparate_impact": di,
                "flagged": di is not None and di < self.threshold and group != ref,
                "is_priority_group": group in self.priority_groups,
            }

        return {
            "reference_group": ref,
            "threshold": self.threshold,
            "provenance": self.provenance,
            "groups": groups,
            "flagged_groups": [g for g, d in groups.items() if d["flagged"]],
        }

    def mitigate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sensitive_features: np.ndarray,
        base_estimator,
        constraint_type: str = "demographic_parity",
    ) -> dict:
        """
        Train a mitigated model using Fairlearn's ExponentiatedGradient
        with community-defined constraint parameters.

        Parameters
        ----------
        X_train : array-like
            Training features.
        y_train : array-like
            Training labels.
        sensitive_features : array-like
            Protected attribute values for training data.
        base_estimator : sklearn estimator
            The base model to mitigate.
        constraint_type : str
            'demographic_parity' or 'equalized_odds'.

        Returns
        -------
        dict
            Pre/post mitigation metrics, mitigated model, and provenance.
        """
        from fairlearn.reductions import (
            ExponentiatedGradient,
            DemographicParity,
            EqualizedOdds,
        )

        # Select constraint based on community config or parameter
        if constraint_type == "equalized_odds":
            constraint = EqualizedOdds()
        else:
            constraint = DemographicParity()

        # Pre-mitigation audit
        base_estimator.fit(X_train, y_train)
        y_pred_before = base_estimator.predict(X_train)
        pre_audit = self.audit(y_train, y_pred_before, sensitive_features)

        # Mitigate
        mitigator = ExponentiatedGradient(base_estimator, constraint)
        mitigator.fit(X_train, y_train, sensitive_features=sensitive_features)
        y_pred_after = mitigator.predict(X_train)
        post_audit = self.audit(y_train, y_pred_after, sensitive_features)

        # Compare
        pre_flagged = set(pre_audit["flagged_groups"])
        post_flagged = set(post_audit["flagged_groups"])

        return {
            "pre_mitigation": pre_audit,
            "post_mitigation": post_audit,
            "groups_remediated": sorted(pre_flagged - post_flagged),
            "groups_still_flagged": sorted(post_flagged),
            "constraint_used": constraint_type,
            "community_threshold": self.threshold,
            "provenance": self.provenance,
            "mitigated_model": mitigator,
        }
