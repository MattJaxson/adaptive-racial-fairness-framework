def reweight_samples_with_community(data, race_col, outcome_col, favorable, community_defs):
    """
    Reweight samples based on community-defined priority groups and fairness target.
    """
    target_group = community_defs.get('fairness_target', 'White')
    priority_groups = community_defs.get('priority_groups', [])

    outcome_rates = data.groupby(race_col)[outcome_col].value_counts(normalize=True).unstack().fillna(0)
    weights = [1.0] * len(data)

    for idx, row in data.iterrows():
        race = row[race_col]
        if race in priority_groups:
            group_rate = outcome_rates.loc[race, favorable] if race in outcome_rates.index else 0.5
            target_rate = outcome_rates.loc[target_group, favorable] if target_group in outcome_rates.index else 0.5
            if group_rate == 0:
                weight = 1.0
            else:
                weight = target_rate / (group_rate + 1e-6)
            weights[idx] = weight

    data['sample_weight'] = weights
    print("Applied community-driven reweighting. Target group for parity:", target_group)
    return data
