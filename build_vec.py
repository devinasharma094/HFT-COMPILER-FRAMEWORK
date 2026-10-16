"""
build_vec.py

Builds the Validity Envelope Certificate (VEC) for a trained XGBoost model,
using the held-out calibration split produced by train_model.py.

For each volatility regime (calm / normal / volatile):
  1. Compute a split-conformal interval per feature at a stated coverage
     level (e.g. 90%) -- this is the certified feature-space region.
  2. Walk every split node in every tree of the model. For each split,
     restrict the calibration set to rows (a) in this regime and (b) inside
     the certified interval for the feature that split tests, then compute
     the empirical firing rate (fraction of those rows that go left, i.e.
     satisfy the split condition) and a Clopper-Pearson upper confidence
     bound on it.

Output: vec.json -- one record per (regime, feature) interval clause, plus
one record per (regime, tree, node) branch-firing-rate clause. This is the
artifact Phase 2 (the compiler pass) consumes.

Usage:
    python build_vec.py --model model.json --calib calibration.parquet --out vec.json
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBClassifier

FEATURE_COLS = ["realized_vol", "trade_intensity", "avg_trade_size"]
REGIMES = ["calm", "normal", "volatile"]

COVERAGE_LEVEL = 0.90          # conformal coverage for feature intervals
CONFIDENCE_LEVEL = 0.95        # confidence level for branch-firing upper bounds
MIN_CLAUSE_SUPPORT = 30        # minimum calibration rows required to certify a clause


def conformal_interval(values: pd.Series, coverage: float) -> tuple[float, float]:
    """Split-conformal-style interval: the symmetric central quantile range
    that contains `coverage` fraction of the calibration values for this
    feature in this regime. This is the certified feature-space region.
    """
    alpha = 1 - coverage
    lower = values.quantile(alpha / 2)
    upper = values.quantile(1 - alpha / 2)
    return float(lower), float(upper)


def clopper_pearson_upper(successes: int, trials: int, confidence: float) -> float:
    """Upper confidence bound on a binomial proportion (Clopper-Pearson /
    exact method). Used to upper-bound each split's branch-firing rate.
    """
    if trials == 0:
        return 1.0  # no data -> cannot certify, treat as unbounded (caller should drop)
    if successes == trials:
        return 1.0
    alpha = 1 - confidence
    upper = stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return float(upper)


def build_feature_intervals(calib: pd.DataFrame) -> dict:
    """Per-regime, per-feature certified intervals."""
    clauses = {}
    for regime in REGIMES:
        regime_df = calib[calib["regime"] == regime]
        clauses[regime] = {}
        for feat in FEATURE_COLS:
            if len(regime_df) < MIN_CLAUSE_SUPPORT:
                continue
            lower, upper = conformal_interval(regime_df[feat], COVERAGE_LEVEL)
            clauses[regime][feat] = {
                "lower_bound": lower,
                "upper_bound": upper,
                "coverage_level": COVERAGE_LEVEL,
                "n_calibration_rows": int(len(regime_df)),
            }
    return clauses


def restrict_to_envelope(df: pd.DataFrame, regime: str, feature_clauses: dict) -> pd.DataFrame:
    """Rows in this regime AND inside every certified feature interval for that regime."""
    out = df[df["regime"] == regime]
    for feat, clause in feature_clauses.get(regime, {}).items():
        out = out[(out[feat] >= clause["lower_bound"]) & (out[feat] <= clause["upper_bound"])]
    return out


def build_branch_clauses(model: XGBClassifier, calib: pd.DataFrame, feature_clauses: dict) -> list:
    """Walk every split node in every tree; compute per-regime, in-envelope
    firing rate and its upper confidence bound.
    """
    booster = model.get_booster()
    tree_df = booster.trees_to_dataframe()
    # tree_df columns include: Tree, Node, ID, Feature, Split, Yes, No, ...
    split_nodes = tree_df[tree_df["Feature"] != "Leaf"].copy()

    print(f"  {len(split_nodes):,} split nodes across {split_nodes['Tree'].nunique()} trees")

    branch_clauses = []
    for regime in REGIMES:
        envelope_df = restrict_to_envelope(calib, regime, feature_clauses)
        n_envelope = len(envelope_df)
        if n_envelope < MIN_CLAUSE_SUPPORT:
            continue

        # Pre-sort each feature column once; "number of rows with x < t" is then a binary search.
        sorted_cols = {f: np.sort(envelope_df[f].to_numpy()) for f in FEATURE_COLS}

        for row in split_nodes.itertuples(index=False):
            feat = row.Feature
            threshold = row.Split
            if feat not in FEATURE_COLS:
                continue  # split on a feature outside our certified set -> cannot certify

            fires_left = int(np.searchsorted(sorted_cols[feat], threshold, side="left"))
            upper_bound = clopper_pearson_upper(fires_left, n_envelope, CONFIDENCE_LEVEL)

            branch_clauses.append(
                {
                    "regime": regime,
                    "tree_index": int(row.Tree),
                    "node_id": row.ID,
                    "feature": feat,
                    "threshold": float(threshold),
                    "n_envelope_rows": int(n_envelope),
                    "firing_rate_upper_bound": upper_bound,
                    "confidence_level": CONFIDENCE_LEVEL,
                }
            )

    return branch_clauses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model.json")
    parser.add_argument("--calib", required=True, help="Path to calibration.parquet")
    parser.add_argument("--out", default="vec.json", help="Output VEC file")
    parser.add_argument("--coverage", type=float, default=0.90,
                        help="per-feature interval coverage level (higher = wider boxes, fewer fallbacks, fewer eliminations)")
    args = parser.parse_args()
    global COVERAGE_LEVEL
    COVERAGE_LEVEL = args.coverage

    print(f"Loading model from {args.model} ...")
    model = XGBClassifier()
    model.load_model(args.model)

    print(f"Loading calibration data from {args.calib} ...")
    calib = pd.read_parquet(args.calib)
    print(f"  {len(calib):,} rows")
    print(f"  Regime breakdown:\n{calib['regime'].value_counts()}")

    print("Building per-regime feature intervals (conformal calibration) ...")
    feature_clauses = build_feature_intervals(calib)
    for regime, feats in feature_clauses.items():
        print(f"  [{regime}]")
        for feat, clause in feats.items():
            print(
                f"    {feat}: [{clause['lower_bound']:.6g}, {clause['upper_bound']:.6g}] "
                f"(n={clause['n_calibration_rows']:,})"
            )

    print("Building branch-firing-rate clauses (this can take a few minutes) ...")
    branch_clauses = build_branch_clauses(model, calib, feature_clauses)
    print(f"  Built {len(branch_clauses):,} branch clauses")

    vec = {
        "schema_version": 1,
        "coverage_level": COVERAGE_LEVEL,
        "confidence_level": CONFIDENCE_LEVEL,
        "feature_intervals": feature_clauses,
        "branch_clauses": branch_clauses,
    }

    with open(args.out, "w") as f:
        json.dump(vec, f, indent=2)
    print(f"Saved VEC to {args.out}")


if __name__ == "__main__":
    main()
