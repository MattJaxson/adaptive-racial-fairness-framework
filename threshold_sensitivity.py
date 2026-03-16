"""
Threshold Sensitivity Analysis
===============================
Sweeps DI thresholds from 0.70 to 0.95 across all datasets and shows how
the number of flagged groups changes. Produces both console output and a
JSON artifact for publication.

This answers the counter-argument: "You just picked 0.9 to flag more people."
The response: "Here is the full landscape. The community picks their point on it,
and we show what that choice means in concrete terms."

Usage:
    python threshold_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


DATASETS = {
    "HR Hiring": {
        "path": "data/real_hr_data.csv",
        "race_col": "race",
        "outcome_col": "hired",
        "favorable": "Yes",
    },
    "HMDA Lending (Michigan)": {
        "path": "data/external/hmda_michigan_lending.csv",
        "race_col": "derived_race",
        "outcome_col": "action_taken",
        "favorable": 1,
    },
    "COMPAS Recidivism": {
        "path": "data/external/compas_recidivism.csv",
        "race_col": "race",
        "outcome_col": "two_year_recid",
        "favorable": 0,
    },
}

SKIP_LABELS = {
    "Race Not Available", "Free Form Text Only", "Joint",
    "2 or more minority races", "Other",
}

THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def compute_di(df, race_col, outcome_col, favorable):
    """Compute DI ratios for all groups relative to White/Caucasian/highest-rate."""
    binary = (df[outcome_col] == favorable).astype(float)
    group_rates = binary.groupby(df[race_col]).mean().to_dict()

    # Reference group
    if "White" in group_rates:
        ref = "White"
    elif "Caucasian" in group_rates:
        ref = "Caucasian"
    else:
        ref = max(group_rates, key=lambda g: group_rates[g])

    ref_rate = group_rates[ref]

    di_ratios = {}
    for group, rate in group_rates.items():
        if group == ref:
            di_ratios[group] = 1.0
        elif ref_rate == 0:
            di_ratios[group] = None
        else:
            di_ratios[group] = round(rate / ref_rate, 4)

    return di_ratios, group_rates, ref


def main():
    print("=" * 100)
    print("  THRESHOLD SENSITIVITY ANALYSIS")
    print("  How does the choice of fairness threshold change who is protected?")
    print("=" * 100)
    print()

    all_results = {}

    for name, config in DATASETS.items():
        path = Path(PROJECT_ROOT) / config["path"]
        if not path.exists():
            print(f"  SKIP: {name}")
            continue

        df = pd.read_csv(path)
        mask = ~df[config["race_col"]].isin(SKIP_LABELS)
        df_clean = df[mask].copy()

        di_ratios, group_rates, ref = compute_di(
            df_clean, config["race_col"], config["outcome_col"], config["favorable"]
        )

        n_records = len(df_clean)
        groups = sorted(di_ratios.keys())

        print(f"{'─' * 100}")
        print(f"  DATASET: {name}  ({n_records:,} records, reference: {ref})")
        print(f"{'─' * 100}")
        print()

        # Header
        header = f"  {'GROUP':<42} {'RATE':>7} {'DI':>7}"
        for t in THRESHOLDS:
            header += f" {'θ=' + str(t):>7}"
        print(header)
        print(f"  {'─' * 42} {'─' * 7} {'─' * 7}" + " ─────── " * len(THRESHOLDS))

        dataset_result = {
            "n_records": n_records,
            "reference_group": ref,
            "groups": {},
            "threshold_sweep": {},
        }

        for group in groups:
            di = di_ratios[group]
            rate = group_rates[group]
            row = f"  {group:<42} {rate:>7.1%} {di if di is not None else 'undef':>7}"

            group_data = {"rate": round(rate, 4), "di": di, "flagged_at": []}

            for t in THRESHOLDS:
                if di is not None and di < t and group != ref:
                    row += f" {'FAIL':>7}"
                    group_data["flagged_at"].append(t)
                else:
                    row += f" {'pass':>7}"
            print(row)
            dataset_result["groups"][group] = group_data

        print()

        # Summary: flagged count at each threshold
        print(f"  {'GROUPS FLAGGED':<42} {'':>7} {'':>7}", end="")
        for t in THRESHOLDS:
            count = sum(
                1 for g, di in di_ratios.items()
                if di is not None and di < t and g != ref
            )
            print(f" {count:>7}", end="")
            dataset_result["threshold_sweep"][str(t)] = {
                "flagged_count": count,
                "flagged_groups": [
                    g for g, di in di_ratios.items()
                    if di is not None and di < t and g != ref
                ],
            }
        print()

        # The "critical transition" — where does a group flip from pass to fail?
        print()
        print("  Critical transitions (threshold where group status changes):")
        for group in groups:
            di = di_ratios[group]
            if di is None or group == ref or di >= max(THRESHOLDS):
                continue
            # Find the lowest threshold that flags this group
            first_flag = None
            for t in THRESHOLDS:
                if di < t:
                    first_flag = t
                    break
            if first_flag:
                print(f"    {group}: DI={di:.4f} → first flagged at θ={first_flag} "
                      f"(passes at θ={THRESHOLDS[THRESHOLDS.index(first_flag) - 1] if THRESHOLDS.index(first_flag) > 0 else 'none'})")

        print()
        all_results[name] = dataset_result

    # Overall summary
    print("=" * 100)
    print("  SUMMARY: Total groups flagged across all datasets at each threshold")
    print("=" * 100)
    print()
    print(f"  {'THRESHOLD':<15} {'TOTAL FLAGGED':>15} {'DELTA FROM 0.8':>18}")
    print(f"  {'─' * 15} {'─' * 15} {'─' * 18}")

    baseline_count = 0
    for t in THRESHOLDS:
        total = 0
        for name, result in all_results.items():
            total += result["threshold_sweep"][str(t)]["flagged_count"]
        if t == 0.8:
            baseline_count = total
        delta = total - baseline_count
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        marker = " ← EEOC default" if t == 0.8 else ""
        print(f"  θ = {t:<10} {total:>15} {delta_str:>18}{marker}")

    print()
    print("  KEY INSIGHT: Every 0.05 increase in θ reveals groups that the previous")
    print("  threshold deemed 'fair enough.' The choice of threshold is not neutral —")
    print("  it determines who is visible and who is erased from the audit.")
    print()
    print("=" * 100)

    # Save
    output_path = Path(PROJECT_ROOT) / "docs" / "threshold_sensitivity.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Results saved to {output_path}")


if __name__ == "__main__":
    main()
