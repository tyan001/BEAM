"""Exhaustive feature-subset search for XGBoost on clinical_preprocessed.csv."""
import sys
import warnings
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils import combine_categories, exhaustive_feature_search

DATA_PATH = REPO_ROOT / "data" / "clinical" / "clinical_preprocessed.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
REMOVE_FEATURES = ["CDRGLOB", "AMYLPET", "NACCETPR"]
DROP_COLS = ["PTID", "VISITYR", "FL_UDSD"]

COMBINE_MAP = {"2": ["2", "3"]}
RENUMBER_MAP = {"1": 1, "2": 2, "4": 3, "5": 4, "6": 5}

RESULT_NAME_6CLASS = "xgb_results.csv"
RESULT_NAME_5CLASS = "xgb_results_SCD_Imp.csv"


def make_model():
    return xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=4,
    )


def run_search(df, result_name):
    X = df.drop(columns=DROP_COLS)
    y = df["FL_UDSD"].astype(int) - 1  # xgboost wants 0-indexed int labels
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        results = exhaustive_feature_search(
            X, y,
            groups=df["PTID"],
            cv=cv,
            estimator=make_model(),
            min_features=2,
            max_features=len(X.columns),
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / result_name
    results.to_csv(out, index=False)
    print(f"Wrote {out}")


def main():
    df = pd.read_csv(DATA_PATH)
    df_filter = df.drop(columns=REMOVE_FEATURES).dropna()

    run_search(df_filter, RESULT_NAME_6CLASS)

    df_combined = combine_categories(
        df_filter.assign(FL_UDSD=df_filter["FL_UDSD"].astype(str)),
        combination_map=COMBINE_MAP,
        target_col="FL_UDSD",
    )
    df_combined["FL_UDSD"] = df_combined["FL_UDSD"].map(RENUMBER_MAP).astype(int)
    run_search(df_combined, RESULT_NAME_5CLASS)


if __name__ == "__main__":
    main()
