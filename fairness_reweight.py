import logging

import numpy as np
import pandas as pd


def reweight_samples_with_community(data, race_col, outcome_col, favorable, community_defs):
    """
    Reweight samples based on community-defined priority groups and fairness target.
    Priority group rows are upweighted so their favorable outcome rate matches the target group's.
    """
    target_group = community_defs.get('fairness_target', 'White')
    priority_groups = set(community_defs.get('priority_groups', []))

    outcome_rates = (
        data.groupby(race_col)[outcome_col]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )

    # Resolve per-row group rates via map (vectorized, avoids iterrows)
    has_favorable = favorable in outcome_rates.columns
    group_rate_map = outcome_rates[favorable] if has_favorable else pd.Series(dtype=float)
    target_rate = (
        float(outcome_rates.loc[target_group, favorable])
        if has_favorable and target_group in outcome_rates.index
        else 0.5
    )

    group_rates = data[race_col].map(group_rate_map).fillna(0.5).astype(float).values
    is_priority = data[race_col].isin(priority_groups).values

    # weight = target_rate / group_rate for priority groups; 1.0 otherwise
    with np.errstate(invalid='ignore'):
        raw_weights = np.where(group_rates == 0, 1.0, target_rate / (group_rates + 1e-6))

    weights = np.where(is_priority, raw_weights, 1.0)
    data = data.copy()
    data['sample_weight'] = weights

    logging.info("Reweighting applied. Target group: %s, priority groups: %s", target_group, list(priority_groups))
    return data
