import logging

import pandas as pd


def group_outcomes_by_race(data, race_col, outcome_col):
    """
    Group outcomes by race and normalize proportions.
    """
    if data.empty:
        raise ValueError("Input data is empty.")
    return data.groupby(race_col)[outcome_col].value_counts(normalize=True).unstack().fillna(0)


def disparate_impact(data, race_col, outcome_col, privileged, unprivileged, favorable):
    """
    Calculate disparate impact ratio between privileged and unprivileged groups.
    Returns None if the privileged group has no positive outcomes.
    """
    privileged_rate = data.loc[data[race_col] == privileged, outcome_col].value_counts(normalize=True).get(favorable, 0)
    unprivileged_rate = data.loc[data[race_col] == unprivileged, outcome_col].value_counts(normalize=True).get(favorable, 0)
    if privileged_rate == 0:
        logging.warning(
            "Privileged group '%s' has no positive outcomes — disparate impact is undefined.", privileged
        )
        return None
    return unprivileged_rate / privileged_rate
