#!/usr/bin/env python
"""
Group-aware cross-validated experiments across feature-group combinations
for FL_UDSD diagnostic classification.

Two scenarios are run:
  1. Drop "Impaired Not SCD/MCI" (FL_UDSD == 2) and classify the remaining 4 classes.
  2. Merge "Impaired Not SCD/MCI" (2) into "EMCI" (4) and classify the resulting 4 classes.

Why this is a script and not a notebook:
  The original notebook hung intermittently. The most likely culprit is
  *nested parallelism*: cross_validate(n_jobs=-1) launches loky workers,
  and each XGBoost (tree_method='hist') / RandomForest worker spawns its
  own thread pool. On many-core machines this oversubscribes the CPU and
  loky workers can deadlock or appear hung. We fix this by:
    - capping BLAS / OMP / MKL thread counts BEFORE importing numpy/sklearn
    - exposing CV_N_JOBS as a CLI flag (default 1 for XGB/RF, -1 for LR)
    - using the 'loky' backend explicitly with a sane worker count
    - running each model in its own subprocess via __main__ guard

Usage:
    python logistic_regression_groups.py --model xgb
    python logistic_regression_groups.py --model lr --n-jobs -1
    python logistic_regression_groups.py --model rf --n-jobs 4 --skip-scenario-2
"""

# ---------------------------------------------------------------------------
# Thread-pool caps. These MUST be set before importing numpy / sklearn / xgb.
# Each parallel CV worker would otherwise inherit OMP_NUM_THREADS = ncores
# and the BLAS pools would fight loky for cores -> the classic "hang".
# ---------------------------------------------------------------------------
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import sys
import warnings
from itertools import combinations
from math import floor, log10
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, cross_validate,  StratifiedKFold

import xgboost as xgb
from catboost import CatBoostClassifier
from utils.feature_group import load_feature_groups
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
# MRI_VOLUME = [
#     "VOL_ENTRHNA_L", "VOL_ENTRHNA_R", "VOL_HIPP_L", "VOL_HIPP_R",
#     "VOL_AMYG_L", "VOL_AMYG_R", "VOL_PRECUNE_L", "VOL_PRECUNE_R",
#     "VOL_POST_CING_L", "VOL_POST_CING_R", "VOL_INF_PAR_L", "VOL_INF_PAR_R",
#     "VOL_INF_TEMP_L", "VOL_INF_TEMP_R", "VOL_TEMP_PL_L", "VOL_TEMP_PL_R",
#     "VOL_LAT_ORB_L", "VOL_LAT_ORB_R", "VOL_SUP_FRNT_L", "VOL_SUP_FRNT_R",
#     "VOL_PRECENT_L", "VOL_PRECENT_R",
#     "VOL_HIPP_SBCLM_HEAD_L", "VOL_HIPP_SBCLM_HEAD_R",
#     "VOL_HIPP_SBCLM_BOD_L", "VOL_HIPP_SBCLM_BOD_R",
#     "VOL_HIPP_PRESBCLM_HEAD_L", "VOL_HIPP_PRESBCLM_HEAD_R",
#     "VOL_HIPP_PRESBCLM_BOD_L", "VOL_HIPP_PRESBCLM_BOD_R",
#     "VOL_HIPP_CA1_HEAD_L", "VOL_HIPP_CA1_HEAD_R",
#     "VOL_HIPP_CA1_BOD_L", "VOL_HIPP_CA1_BOD_R",
# ]

# MRI_THICKNESS = [
#     "THK_ENTRHNA_L", "THK_ENTRHNA_R", "THK_PARAHIPP_L", "THK_PARAHIPP_R",
#     "THK_PRECUNE_L", "THK_PRECUNE_R", "THK_POST_CING_L", "THK_POST_CING_R",
#     "THK_INF_PAR_L", "THK_INF_PAR_R", "THK_INF_TEMP_L", "THK_INF_TEMP_R",
#     "THK_TEMP_PL_L", "THK_TEMP_PL_R", "THK_LAT_ORB_L", "THK_LAT_ORB_R",
#     "THK_ROST_MIDFRNT_L", "THK_ROST_MIDFRNT_R",
#     "THK_CAUD_MIDFRNT_L", "THK_CAUD_MIDFRNT_R",
#     "THK_SUP_FRNT_L", "THK_SUP_FRNT_R", "THK_PRECENT_L", "THK_PRECENT_R",
# ]

# FEATURE_GROUPS = {
#     "INFO":           ["PTID", "VISITYR"],
#     "CDRSUM":         ["CDRSUM"],
#     "MMSE":           ["MMSE"],
#     "HVLTDR":         ["HVLT_DR"],
#     "LASSI":          ["LASSI_A_CR2", "LASSI_A_CR2_INT","LASSI_B_CR1", "LASSI_B_CR1_INT", "LASSI_B_CR2", "LASSI_B_CR2_INT"],
#     "PLASMA":         ["PTAU_217_CONCNTRTN"],
#     "APOE":           ["APOE4S"],
#     "MRI":            MRI_VOLUME + MRI_THICKNESS,
#     "TARGETS":        ["FL_UDSD", "NACCETPR"],
# }

FEATURE_GROUPS = load_feature_groups()

# Groups eligible to participate in the exhaustive combination search
X_FEATURE_GROUP = {
    k: v for k, v in FEATURE_GROUPS.items() if k not in ("INFO", 'CLINICAL', "TARGETS")
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sigfig(x, n=4):
    x = float(x)
    if x == 0:
        return 0.0
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))


def build_experiments(group_dict):
    """Generate every non-empty combination of feature groups."""
    experiments = {}
    exp_num = 1
    for r in range(1, len(group_dict) + 1):
        for combo in combinations(group_dict.items(), r):
            names = [name for name, _ in combo]
            features = [c for _, feats in combo for c in feats]
            experiments[f"E{exp_num}_{'_'.join(names)}"] = features
            exp_num += 1
    return experiments


def fold_class_counts(y, groups, cv, label_map=None):
    def relabel(k):
        if label_map is not None and k in label_map:
            return label_map[k]
        return int(k) if float(k).is_integer() else k

    train_counts, test_counts = [], []
    for tr_idx, te_idx in cv.split(pd.DataFrame(index=y.index), y, groups):
        tr = y.iloc[tr_idx].value_counts().sort_index().to_dict()
        te = y.iloc[te_idx].value_counts().sort_index().to_dict()
        train_counts.append({relabel(k): int(v) for k, v in tr.items()})
        test_counts.append({relabel(k): int(v) for k, v in te.items()})
    return train_counts, test_counts


def build_scoring(classes, label_map=None):
    """f1_macro + balanced_accuracy + per-class precision/recall."""
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


def make_model(model_name):
    if model_name == "lr":
        return LogisticRegression(
            max_iter=5000, random_state=42, class_weight="balanced",
            n_jobs=1,  # keep solver single-threaded; parallelism lives at CV layer
        )
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced",
            n_jobs=1,  # avoid nested parallelism
        )
    if model_name == "xgb":
        return xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=1,  # CRITICAL: prevents oversubscription with loky workers
        )
    if model_name == "cat":
        return CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=None,       # auto-tuned by CatBoost based on iterations
            task_type="GPU",
            loss_function="MultiClass",
            eval_metric="TotalF1",    # macro F1; better signal than accuracy for imbalanced data
            auto_class_weights="Balanced",  # inversely weights classes by frequency
            random_seed=42,
            thread_count=1,           # CRITICAL: CatBoost's n_jobs equivalent; keep 1 under loky
            verbose=0,
        )
    raise ValueError(f"Unknown model {model_name!r}")


def run_group_experiments(
    df,
    experiments,
    target_col,
    group_col,
    estimator,
    cv,
    n_jobs=1,
    sigfigs=4,
    label_map=None,
    verbose=True,
):
    y = df[target_col]
    groups = df[group_col]
    classes = sorted(y.unique())
    scoring = build_scoring(classes, label_map)

    train_counts, test_counts = fold_class_counts(y, groups, cv, label_map=label_map)

    results = []
    n_exp = len(experiments)
    # Use loky backend explicitly. Process-based isolation prevents one
    # bad fold from poisoning the others.
    with joblib.parallel_backend("loky", n_jobs=n_jobs):
        for i, (exp_name, features) in enumerate(experiments.items(), start=1):
            X_exp = df[features]

            cv_res = cross_validate(
                estimator, X_exp, y,
                groups=groups,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                return_train_score=False,
                error_score="raise",
            )

            row = {"experiment": exp_name, "n_features": len(features)}
            for metric in scoring:
                scores = cv_res[f"test_{metric}"]
                row[f"{metric}_mean"] = sigfig(scores.mean(), sigfigs)
                row[f"{metric}_std"] = sigfig(scores.std(), sigfigs)
                row[f"{metric}_folds"] = [sigfig(s, sigfigs) for s in scores]

            row[f"{target_col}_train_counts_folds"] = train_counts
            row[f"{target_col}_test_counts_folds"] = test_counts
            row["features"] = features
            results.append(row)

            if verbose:
                print(
                    f"[{i:>3}/{n_exp}] {exp_name:50s} "
                    f"f1_macro={row['f1_macro_mean']:.{sigfigs}g}"
                    f"±{row['f1_macro_std']:.{sigfigs}g}  "
                    f"bal_acc={row['balanced_accuracy_mean']:.{sigfigs}g}"
                    f"±{row['balanced_accuracy_std']:.{sigfigs}g}",
                    flush=True,
                )
                for c in classes:
                    name = label_map[c] if (label_map and c in label_map) else c
                    p = row[f"precision_{name}_mean"]
                    r = row[f"recall_{name}_mean"]
                    print(f"      [{name}]  precision={p:.{sigfigs}g}  recall={r:.{sigfigs}g}",
                          flush=True)

    return (
        pd.DataFrame(results)
          .sort_values("f1_macro_mean", ascending=False)
          .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def scenario_drop_impaired(df, model, cv, experiments, n_jobs, out_path):
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

    results_df = run_group_experiments(
        df_filter, experiments, "FL_UDSD_CAT", "PTID",
        model, cv, n_jobs=n_jobs, label_map=diagnosis_map,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return results_df


def scenario_merge_impaired_emci(df, model, cv, experiments, n_jobs, out_path):
    """Merge FL_UDSD == 2 into 4 (EMCI), classify resulting 4 classes."""
    print("\n" + "=" * 72)
    print("SCENARIO 2: Merge 'Impaired Not SCD/MCI' into EMCI")
    print("=" * 72)

    df2 = df.copy()
    df2.loc[df2["FL_UDSD"] == 2, "FL_UDSD"] = 4
    mapping = {old: new for new, old in enumerate(sorted(df2["FL_UDSD"].unique()))}
    df2["FL_UDSD_CAT"] = df2["FL_UDSD"].map(mapping)

    print("Class counts:")
    print(df2[["FL_UDSD", "FL_UDSD_CAT"]].value_counts().sort_index())

    diagnosis_order = ["Normal cognition", "SCD",
                       "Early Impaired", "Late MCI", "Dementia"]
    diagnosis_map = {i: label for i, label in enumerate(diagnosis_order)}

    df2 = df2.drop(columns=["NACCETPR", "FL_UDSD"])

    results_df = run_group_experiments(
        df2, experiments, "FL_UDSD_CAT", "PTID",
        model, cv, n_jobs=n_jobs, label_map=diagnosis_map,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return results_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["lr", "rf", "xgb", "cat"], default="xgb",
                   help="Estimator to use (default: xgb)")
    p.add_argument("--data",
                   help="Path to preprocessed clinical CSV")
    p.add_argument("--results-dir", default="results",
                   help="Directory to write CSV results into (default: results)")
    p.add_argument("--n-jobs", type=int, default=None,
                   help="CV parallel jobs. Default: -1 for lr, 2 for rf/xgb. "
                        "Set to 1 if you are still seeing hangs.")
    p.add_argument("--n-splits", type=int, default=4,
                   help="StratifiedGroupKFold splits (default: 5)")
    p.add_argument("--skip-scenario-1", action="store_true",
                   help="Skip the 'drop Impaired Not SCD/MCI' run")
    p.add_argument("--skip-scenario-2", action="store_true",
                   help="Skip the 'merge into EMCI' run")
    return p.parse_args()


def main():
    args = parse_args()

    # Sensible default n_jobs per model. lr is light and benefits from
    # full parallelism; xgb/rf are heavy per-fit and too many workers
    # is what tended to hang the notebook.
    if args.n_jobs is None:
        args.n_jobs = 2 if args.model in ("xgb", "cat") else -1

    print(f"Model        : {args.model}")
    print(f"CV n_jobs    : {args.n_jobs}")
    print(f"Data         : {args.data}")
    print(f"CV splits    : {args.n_splits}")
    print(f"Results dir  : {args.results_dir}")
    print(f"{X_FEATURE_GROUP}")

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"ERROR: data file not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df):,} rows, {df.shape[1]} columns")
    print("FL_UDSD distribution:")
    print(df["FL_UDSD"].value_counts().sort_index())

    experiments = build_experiments(X_FEATURE_GROUP)
    print(f"\nGenerated {len(experiments)} feature-group experiments")

    # cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    model = make_model(args.model)

    results_dir = Path(args.results_dir)

    if not args.skip_scenario_1:
        scenario_drop_impaired(
            df, model, cv, experiments, args.n_jobs,
            results_dir / f"{args.model}_results.csv",
        )

    if not args.skip_scenario_2:
        scenario_merge_impaired_emci(
            df, model, cv, experiments, args.n_jobs,
            results_dir / f"{args.model}_results_early_impaired.csv",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()