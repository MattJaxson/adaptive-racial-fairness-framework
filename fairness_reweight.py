"""
Fairness Reweighting
---------------------
Assigns per-sample weights to align priority group outcome rates toward
the reference (target) group rate, as defined in the community configuration.

Algorithm (from algorithm_specification.md Section 4):

    For each individual i in priority group g:
        If yᵢ = 1 (favorable):  wᵢ = target_rate / P(Y=1 | A=g)
        If yᵢ = 0 (unfavorable): wᵢ = (1 - target_rate) / (1 - P(Y=1 | A=g))

    For individuals not in a priority group:
        wᵢ = 1.0
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def reweight_samples_with_community(data, race_col, outcome_col, favorable, community_defs):
    """
    Reweight samples based on community-defined priority groups and fairness target.

    Raises ValueError if target group is missing from data or has no favorable outcomes,
    rather than silently falling back — community-defined parameters must be validated.
    """
    target_group = community_defs.get('fairness_target', 'White')
    priority_groups = set(community_defs.get('priority_groups', []))

    # --- Validate target group is in the data ---
    groups_in_data = set(data[race_col].dropna().unique())
    if target_group not in groups_in_data:
        available = sorted(groups_in_data)
        raise ValueError(
            f"fairness_target '{target_group}' not found in data. "
            f"Available groups: {available}. "
            f"Update community_definitions.json or pass a valid privileged_group."
        )

    # --- Compute per-group favorable outcome rates ---
    binary_outcome = (data[outcome_col] == favorable).astype(float)
    group_rates = binary_outcome.groupby(data[race_col]).mean()

    target_rate = float(group_rates.get(target_group, 0.0))
    if target_rate == 0.0:
        raise ValueError(
            f"Target group '{target_group}' has a 0% favorable outcome rate. "
            f"Cannot reweight toward a group with no favorable outcomes."
        )
    if target_rate == 1.0:
        logger.warning(
            "Target group '%s' has a 100%% favorable outcome rate. "
            "Reweighting will push all priority group weights toward certainty.",
            target_group,
        )

    # --- Warn about missing priority groups ---
    missing_priority = priority_groups - groups_in_data
    if missing_priority:
        logger.warning(
            "Priority group(s) %s not found in data — they will be skipped.",
            sorted(missing_priority),
        )

    # --- Compute weights per row ---
    data = data.copy()
    is_favorable = binary_outcome.values.astype(bool)
    row_group = data[race_col].values
    row_rates = data[race_col].map(group_rates).values.astype(float)

    # Per the algorithm spec:
    #   favorable:   w = target_rate / group_rate
    #   unfavorable: w = (1 - target_rate) / (1 - group_rate)
    # Only applied to priority groups; everyone else gets weight 1.0.
    weights = np.ones(len(data), dtype=float)

    for group in priority_groups & groups_in_data:
        group_mask = row_group == group
        g_rate = float(group_rates[group])

        if g_rate == 0.0:
            # Group has 0% favorable rate — can only reweight unfavorable outcomes
            fav_weight = 1.0
            unfav_weight = (1.0 - target_rate) / 1.0  # denominator is (1 - 0) = 1
        elif g_rate == 1.0:
            # Group has 100% favorable rate — can only reweight favorable outcomes
            fav_weight = target_rate / 1.0
            unfav_weight = 1.0
        else:
            fav_weight = target_rate / g_rate
            unfav_weight = (1.0 - target_rate) / (1.0 - g_rate)

        fav_mask = group_mask & is_favorable
        unfav_mask = group_mask & ~is_favorable
        weights[fav_mask] = fav_weight
        weights[unfav_mask] = unfav_weight

    data['sample_weight'] = weights

    logger.info(
        "Reweighting applied. Target group: %s (rate=%.4f), priority groups: %s",
        target_group, target_rate, sorted(priority_groups & groups_in_data),
    )
    return data
