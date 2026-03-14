# racial_bias_score.py

import pandas as pd
import numpy as np


def calculate_racial_bias_score(df, sensitive_column='race', outcome_column='outcome'):
    """
    Calculate a basic racial bias score based on outcome disparities across racial groups.
    Returns a dictionary of group outcomes and a disparity score (max - min outcome rate).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with at least the sensitive_column and outcome_column.
    sensitive_column : str
        Column containing racial/ethnic group labels.
    outcome_column : str
        Column containing numeric binary outcomes (0/1 or float 0.0-1.0).

    Raises
    ------
    ValueError
        If required columns are missing, outcome is not numeric, or data is empty.
    """
    # Validate columns exist
    for col in (sensitive_column, outcome_column):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )

    if df.empty:
        raise ValueError("Dataset is empty.")

    # Validate outcome column is numeric
    if not pd.api.types.is_numeric_dtype(df[outcome_column]):
        raise ValueError(
            f"Outcome column '{outcome_column}' must be numeric (0/1). "
            f"Got dtype: {df[outcome_column].dtype}"
        )

    # Drop rows with NaN in either column and warn
    clean = df[[sensitive_column, outcome_column]].dropna()
    n_dropped = len(df) - len(clean)
    if n_dropped > 0:
        import logging
        logging.getLogger(__name__).warning(
            "%d rows dropped due to missing values in '%s' or '%s'.",
            n_dropped, sensitive_column, outcome_column,
        )

    if clean.empty:
        raise ValueError("No valid rows after dropping missing values.")

    group_stats = clean.groupby(sensitive_column)[outcome_column].mean()

    # Handle single-group edge case
    if len(group_stats) < 2:
        return {
            "group_outcomes": group_stats.to_dict(),
            "racial_disparity_score": 0.0,
        }

    disparity = group_stats.max() - group_stats.min()

    return {
        "group_outcomes": group_stats.to_dict(),
        "racial_disparity_score": round(float(disparity), 4),
    }
