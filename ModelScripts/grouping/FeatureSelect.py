#!/usr/bin/env python
"""
Feature-selection pipeline for FL_UDSD diagnostic classification.

Selects features from the full feature pool inside cross-validation
(no data leakage) and trains a classifier on the selected subset.

Feature selection methods:
  rfecv       : Recursive Feature Elimination with inner CV to pick the
                optimal number of features automatically.
  selectkbest : Filter by ANOVA F-score, keep top-k features (set with --k).

RFECV uses a lightweight ranker (LR for lr models, RF-50 for tree/GPU
models) to avoid nested-parallelism hangs from running the full classifier
inside the inner CV on every fold.

Drops "Impaired Not SCD/MCI" (FL_UDSD == 2) and classifies the remaining 4 classes.

Usage:
    python FeatureSelect.py --model xgb --fs-method rfecv --data path/to/data.csv
    python FeatureSelect.py --model lr  --fs-method selectkbest --k 15 --data ...
    python FeatureSelect.py --model rf  --fs-method rfecv --n-splits 4 --data ...
"""

# ---------------------------------------------------------------------------
# Thread-pool caps — must be set before importing numpy / sklearn / xgb.
# ---------------------------------------------------------------------------
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import warnings
from math import floor, log10
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline

import xgboost as xgb
from catboost import CatBoostClassifier
from utils.feature_group import load_feature_groups

warnings.filterwarnings("ignore", category=UserWarning)


FEATURE_GROUPS = load_feature_groups()

# All features eligible for selection (INFO / CLINICAL / TARGETS are metadata)
ALL_FEATURES = [
    feat
    for k, feats in FEATURE_GROUPS.items()
    if k not in ("INFO", "CLINICAL", "TARGETS")
    for feat in feats
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sigfig(x, n=4):
    x = float(x)
    if x == 0:
        return 0.0
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))


def build_scoring(classes, label_map=None):
    scoring = {
        "f1_macro": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
    }
    for c in classes:
        name = label_map[c] if (label_map and c in label_map) else c
        scoring[f"precision_{name}"] = make_scorer(
            precision_score, labels=[c], average="macro", zero_division=0
        )
        scoring[f"recall_{name}"] = make_scorer(
            recall_score, labels=[c], average="macro", zero_division=0
        )
    return scoring


def make_classifier(model_name):
    if model_name == "lr":
        return LogisticRegression(
            max_iter=5000, random_state=42, class_weight="balanced", n_jobs=1,
        )
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced", n_jobs=1,
        )
    if model_name == "xgb":
        return xgb.XGBClassifier(
            n_estimators=100, random_state=42, objective="multi:softprob",
            eval_metric="mlogloss", tree_method="hist", n_jobs=1,
        )
    if model_name == "catb":
        return CatBoostClassifier(
            iterations=300, depth=6, learning_rate=None, task_type="GPU",
            loss_function="MultiClass", eval_metric="TotalF1",
            auto_class_weights="Balanced", random_seed=42, thread_count=1, verbose=0,
        )
    raise ValueError(f"Unknown model {model_name!r}")


def resolve_min_features(min_features, n_features):
    """Convert --min-features to an absolute count.

    A value in (0, 1) is treated as a fraction of n_features.
    A value >= 1 is used directly as an integer count.
    """
    if 0 < min_features < 1:
        return max(1, int(round(min_features * n_features)))
    return max(1, int(min_features))


def make_selector(fs_method, model_name, k, n_splits, min_features=1):
    """Build a feature selector.

    RFECV internals:
      - Uses a lightweight ranker (not the main classifier) to keep wall-time
        manageable.  LR is used for lr models; a 50-tree RF is used otherwise.
      - Inner CV has at most 3 folds to limit the nested-CV overhead.
    SelectKBest internals:
      - Uses ANOVA F-score (f_classif), appropriate for continuous features and
        a categorical target.
    """
    if fs_method == "rfecv":
        if model_name == "lr":
            ranker = LogisticRegression(
                max_iter=2000, random_state=42, class_weight="balanced", n_jobs=1,
            )
        else:
            # Lightweight RF — gives feature_importances_ without GPU/nested issues
            ranker = RandomForestClassifier(
                n_estimators=50, random_state=42, class_weight="balanced", n_jobs=1,
            )
        inner_cv = StratifiedKFold(
            n_splits=min(3, n_splits), shuffle=True, random_state=42,
        )
        return RFECV(
            estimator=ranker,
            cv=inner_cv,
            scoring="f1_macro",
            min_features_to_select=min_features,
            step=1,
            n_jobs=1,
        )
    if fs_method == "selectkbest":
        return SelectKBest(score_func=f_classif, k=k)
    raise ValueError(f"Unknown fs_method {fs_method!r}")


def get_selected_features(selector, feature_names):
    """Return the feature names that were kept by the selector after fitting."""
    mask = selector.get_support()
    return [f for f, m in zip(feature_names, mask) if m]


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------
def run_feature_selection(
    df,
    features,
    target_col,
    model_name,
    fs_method,
    k,
    n_splits,
    min_features,
    cv,
    n_jobs,
    label_map,
    out_path,
    sigfigs=4,
):
    """Run FS + classifier pipeline with CV, then refit on full data for reporting."""
    X = df[features]
    y = df[target_col]
    classes = sorted(y.unique())
    scoring = build_scoring(classes, label_map)

    selector = make_selector(fs_method, model_name, k, n_splits, min_features)
    clf = make_classifier(model_name)
    pipeline = Pipeline([("selector", selector), ("clf", clf)])

    print(f"  Running {fs_method.upper()} + {model_name.upper()} "
          f"on {len(features)} input features...")

    with joblib.parallel_backend("loky", n_jobs=n_jobs):
        cv_res = cross_validate(
            pipeline, X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            return_train_score=False,
            error_score="raise",
        )

    row = {
        "fs_method": fs_method,
        "model": model_name,
        "n_input_features": len(features),
    }
    for metric in scoring:
        scores = cv_res[f"test_{metric}"]
        row[f"{metric}_mean"] = sigfig(scores.mean(), sigfigs)
        row[f"{metric}_std"] = sigfig(scores.std(), sigfigs)
        row[f"{metric}_folds"] = [sigfig(s, sigfigs) for s in scores]

    # Refit on the full dataset to determine which features were selected.
    # This is a post-hoc inspection step — CV scores already come from held-out folds.
    refit_pipeline = Pipeline([
        ("selector", make_selector(fs_method, model_name, k, n_splits, min_features)),
        ("clf", make_classifier(model_name)),
    ])
    refit_pipeline.fit(X, y)
    selected = get_selected_features(refit_pipeline.named_steps["selector"], features)
    row["n_selected_features"] = len(selected)
    row["selected_features"] = selected

    # Print summary
    n = sigfigs
    print(f"  f1_macro          = {row['f1_macro_mean']:.{n}g} ± {row['f1_macro_std']:.{n}g}")
    print(f"  balanced_accuracy = {row['balanced_accuracy_mean']:.{n}g} ± {row['balanced_accuracy_std']:.{n}g}")
    print(f"  n_selected        = {len(selected)}")
    print(f"  selected_features = {selected}")
    for c in classes:
        name = label_map[c] if (label_map and c in label_map) else c
        p = row[f"precision_{name}_mean"]
        r = row[f"recall_{name}_mean"]
        print(f"    [{name}]  precision={p:.{n}g}  recall={r:.{n}g}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def scenario_drop_impaired(df, args, cv, out_dir):
    """Drop FL_UDSD == 2 (Impaired Not SCD/MCI), classify remaining 4 classes."""
    print("\n" + "=" * 72)
    print("SCENARIO 1: Drop 'Impaired Not SCD/MCI' (FL_UDSD == 2)")
    print("=" * 72)

    new_df = df[df["FL_UDSD"] != 2].copy()
    mapping = {old: new for new, old in enumerate(sorted(new_df["FL_UDSD"].unique()))}
    new_df["FL_UDSD_CAT"] = new_df["FL_UDSD"].map(mapping)

    print("Class counts:")
    print(new_df[["FL_UDSD", "FL_UDSD_CAT"]].value_counts().sort_index())

    diagnosis_order = ["Normal cognition", "Subjective Cognitive Decline",
                       "Early MCI", "Late MCI", "Dementia"]
    diagnosis_map = {i: label for i, label in enumerate(diagnosis_order)}

    df_filter = new_df.drop(columns=["NACCETPR", "FL_UDSD"])
    features = [f for f in ALL_FEATURES if f in df_filter.columns]

    min_features = resolve_min_features(args.min_features, len(features))
    return run_feature_selection(
        df_filter, features, "FL_UDSD_CAT",
        args.model, args.fs_method, args.k, args.n_splits, min_features,
        cv, args.n_jobs, diagnosis_map,
        out_dir / f"{args.model}_{args.fs_method}_results.csv",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", choices=["lr", "rf", "xgb", "catb"], default="xgb",
                   help="Classifier trained on selected features (default: xgb)")
    p.add_argument("--fs-method", choices=["rfecv", "selectkbest"], default="rfecv",
                   help="Feature selection method (default: rfecv)")
    p.add_argument("--k", type=int, default=10,
                   help="Number of top features for selectkbest (default: 10; "
                        "ignored by rfecv, which chooses k automatically)")
    p.add_argument("--min-features", type=float, default=1,
                   help="Minimum features for rfecv to keep. An integer >= 1 is "
                        "used directly; a float in (0, 1) is treated as a fraction "
                        "of the input feature count (e.g. 0.1 = 10%%). Default: 1.")
    p.add_argument("--data", required=True,
                   help="Path to preprocessed clinical CSV")
    p.add_argument("--results-dir", default="results",
                   help="Directory to write results CSV (default: results)")
    p.add_argument("--n-jobs", type=int, default=None,
                   help="Outer CV parallel jobs. Default: -1 for lr, 2 for others.")
    p.add_argument("--n-splits", type=int, default=4,
                   help="StratifiedKFold splits (default: 4)")
    p.add_argument("--skip-scenario-1", action="store_true",
                   help="Skip the 'drop Impaired Not SCD/MCI' run")
    return p.parse_args()


def main():
    args = parse_args()

    if args.n_jobs is None:
        args.n_jobs = 2 if args.model in ("xgb", "catb", "rf") else -1

    print(f"Model         : {args.model}")
    print(f"FS method     : {args.fs_method}")
    if args.fs_method == "selectkbest":
        print(f"k             : {args.k}")
    if args.fs_method == "rfecv" and args.min_features != 1:
        print(f"min-features  : {args.min_features}")
    print(f"CV n_jobs     : {args.n_jobs}")
    print(f"CV splits     : {args.n_splits}")
    print(f"Data          : {args.data}")
    print(f"Results dir   : {args.results_dir}")
    print(f"Input features ({len(ALL_FEATURES)}): {ALL_FEATURES}")

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"ERROR: data file not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df):,} rows, {df.shape[1]} columns")
    print("FL_UDSD distribution:")
    print(df["FL_UDSD"].value_counts().sort_index())

    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    out_dir = Path(args.results_dir)

    if not args.skip_scenario_1:
        scenario_drop_impaired(df, args, cv, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
