#!/usr/bin/env python3
"""
Reproducibility Runner
=======================
Runs the complete validation pipeline and verifies that results match
published findings. Any researcher can clone the repo and run:

    python reproduce.py

If all checks pass, the published results are independently verified.
If any check fails, the output shows exactly what diverged.

This is the "trust but verify" layer — it proves the findings aren't
cherry-picked or hand-tuned.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PUBLISHED_RESULTS = PROJECT_ROOT / "docs" / "validation_results.json"
PUBLISHED_SENSITIVITY = PROJECT_ROOT / "docs" / "threshold_sensitivity.json"


def check_datasets_exist() -> bool:
    """Verify all required datasets are present."""
    datasets = [
        "data/real_hr_data.csv",
        "data/external/compas_recidivism.csv",
        "data/external/hmda_michigan_lending.csv",
    ]
    all_present = True
    for ds in datasets:
        path = PROJECT_ROOT / ds
        if path.exists():
            df = pd.read_csv(path)
            print(f"  OK  {ds} ({len(df):,} records)")
        else:
            print(f"  MISSING  {ds}")
            all_present = False
    return all_present


def run_validation_study() -> dict:
    """Re-run the validation study from scratch and return results."""
    from validation_study import DATASETS, RESEARCHER_DEFAULT, COMMUNITY_DEFINED, run_audit

    skip_labels = {
        "Race Not Available", "Free Form Text Only", "Joint",
        "2 or more minority races", "Other",
    }

    results = {}
    for name, config in DATASETS.items():
        path = PROJECT_ROOT / config["path"]
        if not path.exists():
            continue

        df = pd.read_csv(path)
        mask = ~df[config["race_col"]].isin(skip_labels)
        df_clean = df[mask].copy()

        default_result = run_audit(
            df_clean, config["race_col"], config["outcome_col"], config["favorable"],
            ref_group=None, threshold=RESEARCHER_DEFAULT["threshold"],
        )

        community_ref = "Caucasian" if name == "COMPAS Recidivism" else COMMUNITY_DEFINED["fairness_target"]
        community_result = run_audit(
            df_clean, config["race_col"], config["outcome_col"], config["favorable"],
            ref_group=community_ref, threshold=COMMUNITY_DEFINED["threshold"],
        )

        default_flagged = set(default_result["flagged_groups"])
        community_flagged = set(community_result["flagged_groups"])
        newly_flagged = community_flagged - default_flagged

        results[name] = {
            "n_records": len(df_clean),
            "n_groups": df_clean[config["race_col"]].nunique(),
            "default": default_result,
            "community": community_result,
            "newly_flagged": sorted(newly_flagged),
            "delta_flag_count": len(newly_flagged),
        }

    return results


def verify_against_published(fresh: dict, published: dict) -> tuple[int, int]:
    """Compare fresh results against published results. Returns (passed, failed)."""
    passed = 0
    failed = 0

    for dataset_name in published:
        if dataset_name not in fresh:
            print(f"  FAIL  {dataset_name}: not in fresh results")
            failed += 1
            continue

        pub = published[dataset_name]
        frh = fresh[dataset_name]

        # Check record count
        if pub["n_records"] == frh["n_records"]:
            print(f"  OK  {dataset_name}: record count matches ({frh['n_records']})")
            passed += 1
        else:
            print(f"  FAIL  {dataset_name}: record count — published={pub['n_records']}, fresh={frh['n_records']}")
            failed += 1

        # Check default flagged groups
        pub_default = set(pub["default"]["flagged_groups"])
        frh_default = set(frh["default"]["flagged_groups"])
        if pub_default == frh_default:
            print(f"  OK  {dataset_name}: default flagged groups match ({sorted(frh_default)})")
            passed += 1
        else:
            print(f"  FAIL  {dataset_name}: default flagged — published={sorted(pub_default)}, fresh={sorted(frh_default)}")
            failed += 1

        # Check community flagged groups
        pub_community = set(pub["community"]["flagged_groups"])
        frh_community = set(frh["community"]["flagged_groups"])
        if pub_community == frh_community:
            print(f"  OK  {dataset_name}: community flagged groups match ({sorted(frh_community)})")
            passed += 1
        else:
            print(f"  FAIL  {dataset_name}: community flagged — published={sorted(pub_community)}, fresh={sorted(frh_community)}")
            failed += 1

        # Check newly flagged
        pub_new = set(pub["newly_flagged"])
        frh_new = set(frh["newly_flagged"])
        if pub_new == frh_new:
            print(f"  OK  {dataset_name}: newly flagged groups match ({sorted(frh_new)})")
            passed += 1
        else:
            print(f"  FAIL  {dataset_name}: newly flagged — published={sorted(pub_new)}, fresh={sorted(frh_new)}")
            failed += 1

        # Check DI ratios are within tolerance
        for config_type in ("default", "community"):
            pub_di = pub[config_type]["disparate_impact"]
            frh_di = frh[config_type]["disparate_impact"]
            di_match = True
            for group in pub_di:
                if group not in frh_di:
                    di_match = False
                    break
                if pub_di[group] is None and frh_di[group] is None:
                    continue
                if pub_di[group] is None or frh_di[group] is None:
                    di_match = False
                    break
                if abs(float(pub_di[group]) - float(frh_di[group])) > 0.001:
                    di_match = False
                    break

            if di_match:
                print(f"  OK  {dataset_name}: {config_type} DI ratios match (within 0.001)")
                passed += 1
            else:
                print(f"  FAIL  {dataset_name}: {config_type} DI ratios diverge")
                failed += 1

    return passed, failed


def check_core_claim() -> tuple[int, int]:
    """
    Verify the specific published claim:
    'In 67% of datasets, community-defined thresholds flagged additional groups.'
    """
    passed = 0
    failed = 0

    fresh = run_validation_study()
    datasets_with_delta = sum(1 for r in fresh.values() if r["delta_flag_count"] > 0)
    total = len(fresh)
    pct = int(datasets_with_delta / total * 100) if total > 0 else 0

    if datasets_with_delta == 2 and total == 3:
        print(f"  OK  Core claim verified: {datasets_with_delta}/{total} datasets ({pct}%) show delta")
        passed += 1
    else:
        print(f"  FAIL  Core claim: expected 2/3 datasets with delta, got {datasets_with_delta}/{total}")
        failed += 1

    # Verify specific groups
    hmda_new = set(fresh.get("HMDA Lending (Michigan)", {}).get("newly_flagged", []))
    compas_new = set(fresh.get("COMPAS Recidivism", {}).get("newly_flagged", []))

    if "Black or African American" in hmda_new:
        print("  OK  HMDA: 'Black or African American' newly flagged at θ=0.9")
        passed += 1
    else:
        print(f"  FAIL  HMDA: expected 'Black or African American' in newly flagged, got {hmda_new}")
        failed += 1

    if "African-American" in compas_new:
        print("  OK  COMPAS: 'African-American' newly flagged at θ=0.9")
        passed += 1
    else:
        print(f"  FAIL  COMPAS: expected 'African-American' in newly flagged, got {compas_new}")
        failed += 1

    return passed, failed


def main():
    print("=" * 80)
    print("  REPRODUCIBILITY CHECK")
    print("  Verifying published findings can be independently reproduced")
    print("=" * 80)
    print()

    total_passed = 0
    total_failed = 0

    # 1. Check datasets
    print("── Step 1: Dataset Availability ──")
    if not check_datasets_exist():
        print("\n  ABORT: Missing datasets. Cannot reproduce.\n")
        sys.exit(1)
    print()

    # 2. Re-run validation study
    print("── Step 2: Re-running Validation Study ──")
    fresh_results = run_validation_study()
    print(f"  Completed: {len(fresh_results)} datasets processed")
    print()

    # 3. Compare against published
    print("── Step 3: Comparing Against Published Results ──")
    if PUBLISHED_RESULTS.exists():
        with open(PUBLISHED_RESULTS) as f:
            published = json.load(f)
        p, f_ = verify_against_published(fresh_results, published)
        total_passed += p
        total_failed += f_
    else:
        print("  SKIP: No published results to compare against")
    print()

    # 4. Verify core claim
    print("── Step 4: Verifying Core Claim ──")
    p, f_ = check_core_claim()
    total_passed += p
    total_failed += f_
    print()

    # 5. Check schema
    print("── Step 5: Schema Validation ──")
    schema_path = PROJECT_ROOT / "specs" / "community_fairness_config_v1.schema.json"
    if schema_path.exists():
        with open(schema_path) as f:
            schema = json.load(f)
        if schema.get("$id") and schema.get("required"):
            print("  OK  CDF v1.0 schema exists and has required fields")
            total_passed += 1
        else:
            print("  FAIL  Schema missing $id or required fields")
            total_failed += 1
    else:
        print("  FAIL  Schema file not found")
        total_failed += 1
    print()

    # Summary
    print("=" * 80)
    total = total_passed + total_failed
    if total_failed == 0:
        print(f"  ALL CHECKS PASSED ({total_passed}/{total})")
        print()
        print("  The published findings are independently reproducible.")
        print("  Any researcher cloning this repository will get identical results.")
    else:
        print(f"  {total_failed} CHECK(S) FAILED ({total_passed}/{total} passed)")
        print()
        print("  Some findings could not be reproduced. See details above.")
    print("=" * 80)

    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
