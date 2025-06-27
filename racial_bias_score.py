# racial_bias_score.py

import pandas as pd

def calculate_racial_bias_score(df, sensitive_column='race', outcome_column='outcome'):
    """
    Calculate a basic racial bias score based on outcome disparities across racial groups.
    Returns a dictionary of group outcomes and a disparity score.
    """
    group_stats = df.groupby(sensitive_column)[outcome_column].mean()
    disparity = group_stats.max() - group_stats.min()

    return {
        "group_outcomes": group_stats.to_dict(),
        "racial_disparity_score": round(disparity, 4)
    }
