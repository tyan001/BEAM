"""Exhaustive feature-subset search with grouped cross-validation."""
from itertools import combinations

import pandas as pd
from sklearn.model_selection import cross_validate
from tqdm import tqdm


def exhaustive_feature_search(
    X, y, groups, cv, estimator,
    min_features=2, max_features=None,
    scoring=('f1_macro', 'balanced_accuracy'),
    primary_metric='f1_macro',
    n_jobs=-1,
):
    """Evaluate every feature combination with size in [min_features, max_features].

    Iterates over all subsets of the columns of X whose cardinality falls
    within [min_features, max_features] and scores each one with grouped
    cross-validation. Useful for small feature sets where brute-force search
    is tractable; cost grows as the sum of binomial coefficients over the
    requested range, so mind the combinatorial explosion.

    Note on scores: uses sklearn.model_selection.cross_validate, which
    reports held-out TEST scores per fold (not training scores). Each
    combination is scored on every fold's test split and aggregated to
    mean/std, and per-fold scores are also returned.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix. Column names are used to enumerate subsets.
    y : array-like of shape (n_samples,)
        Target vector aligned with X.
    groups : array-like of shape (n_samples,)
        Group labels passed to cv so samples from the same group stay on
        the same side of each split (e.g. patient IDs to prevent leakage).
    cv : cross-validation splitter
        Any splitter accepting groups (e.g. StratifiedGroupKFold).
    estimator : sklearn-compatible estimator
    min_features : int, default=2
        Smallest subset size to evaluate (inclusive).
    max_features : int or None, default=None
        Largest subset size to evaluate (inclusive). None means all
        columns of X.
    scoring : tuple/list/dict of str, default=('f1_macro', 'balanced_accuracy')
        Any scoring strings accepted by sklearn.model_selection.cross_validate.
    primary_metric : str, default='f1_macro'
        Metric used to sort the results. Must appear in `scoring`.
    n_jobs : int, default=-1
        Parallelism passed to cross_validate. -1 uses all cores.

    Returns
    -------
    pandas.DataFrame
        One row per subset, sorted by mean_{primary_metric} descending.
        Columns: n_features, features (list of column names), and for each
        metric m in `scoring`:
          - mean_{m}, std_{m}
          - fold{i}_{m} for i in 0..n_splits-1 (per-fold test score)
    """
    all_features = X.columns.tolist()
    if max_features is None:
        max_features = len(all_features)

    metrics = list(scoring) if not isinstance(scoring, dict) else list(scoring.keys())
    if primary_metric not in metrics:
        raise ValueError(f"primary_metric={primary_metric!r} not in scoring={metrics}")

    results = []
    for k in range(min_features, max_features + 1):
        combos = list(combinations(all_features, k))
        print(f"Evaluating {len(combos)} combinations of size {k}...")
        for feats in tqdm(combos, leave=False):
            feats = list(feats)
            cv_res = cross_validate(
                estimator, X[feats], y,
                groups=groups, cv=cv, scoring=scoring, n_jobs=n_jobs,
                return_train_score=False,
            )
            row = {'n_features': k, 'features': feats}
            for m in metrics:
                fold_scores = cv_res[f'test_{m}']
                row[f'mean_{m}'] = fold_scores.mean()
                row[f'std_{m}'] = fold_scores.std()
                for i, s in enumerate(fold_scores):
                    row[f'fold{i}_{m}'] = s
            results.append(row)

    return (
        pd.DataFrame(results)
          .sort_values(f'mean_{primary_metric}', ascending=False)
          .reset_index(drop=True)
    )
