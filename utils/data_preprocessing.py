"""
Data preprocessing utilities for cognitive diagnosis prediction.

This module contains shared functions for data preprocessing, diagnosis configuration,
and metric calculation used across multiple scripts in the BEAM project.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from typing import Tuple, Dict, List, Optional


def preprocess_data(
    df: pd.DataFrame,
    target_col: str = 'FL_UDSD',
    diagnosis_order: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Preprocess the data by cleaning and encoding features.

    Args:
        df (pd.DataFrame): The input dataframe.
        target_col (str): The target column for diagnosis. Default: 'FL_UDSD'
        diagnosis_order (list): Ordered list of diagnosis categories for encoding.

    Returns:
        pd.DataFrame: The preprocessed dataframe with encoded target variable.
    """
    # Clean data
    filter_df = df[df[target_col] != 'Unknown'].copy()  # Remove rows with unknown target values
    filter_df = filter_df[filter_df["MMSE"] != -1]  # Remove rows with invalid MMSE values

    # Convert columns to categorical if needed
    filter_df['APOE'] = filter_df['APOE'].astype('category')
    filter_df['AMYLPET'] = filter_df['AMYLPET'].astype('category')

    # Encode the target variable as an ordered categorical variable
    if diagnosis_order is not None:
        filter_df['FL_UDSD'] = pd.Categorical(
            filter_df['FL_UDSD'],
            categories=diagnosis_order,
            ordered=True
        )
        filter_df['FL_UDSD_cat'] = filter_df['FL_UDSD'].cat.codes

    return filter_df


def get_diagnosis_config(grouping_strategy: str = "original") -> Tuple[List[str], Optional[Dict[str, List[str]]]]:
    """
    Get diagnosis order and combination map based on grouping strategy.

    Parameters:
    -----------
    grouping_strategy : str
        Strategy for grouping diagnosis categories:
        - "original" : Original 6 categories (no combination)
        - "scd_impaired" : Combine SCD and Impaired Not SCD/MCI
        - "nc_impaired" : Combine Normal Cognition and Impaired Not SCD/MCI

    Returns:
    --------
    tuple: (diagnosis_order, combination_map)
        - diagnosis_order: Ordered list of diagnosis categories
        - combination_map: Dictionary mapping new categories to old categories, or None

    Raises:
    -------
    ValueError: If grouping_strategy is not one of the supported options
    """
    configs = {
        "original": {
            'diagnosis_order': [
                'Normal cognition',
                'Subjective Cognitive Decline',
                'Impaired Not SCD/MCI',
                'Early MCI',
                'Late MCI',
                'Dementia'
            ],
            'combination_map': None
        },
        "scd_impaired": {
            'diagnosis_order': [
                'Normal cognition',
                'SCD/Impaired',
                'Early MCI',
                'Late MCI',
                'Dementia'
            ],
            'combination_map': {
                'SCD/Impaired': ['Subjective Cognitive Decline', 'Impaired Not SCD/MCI']
            }
        },
        "nc_impaired": {
            'diagnosis_order': [
                'NC/Impaired',
                'Subjective Cognitive Decline',
                'Early MCI',
                'Late MCI',
                'Dementia'
            ],
            'combination_map': {
                'NC/Impaired': ['Normal cognition', 'Impaired Not SCD/MCI']
            }
        }
    }

    if grouping_strategy not in configs:
        raise ValueError(
            f"Invalid grouping strategy: '{grouping_strategy}'. "
            f"Must be one of: {list(configs.keys())}"
        )

    config = configs[grouping_strategy]
    return config['diagnosis_order'], config['combination_map']


def combine_categories(
    df: pd.DataFrame,
    combination_map: Optional[Dict[str, List[str]]],
    target_col: str = 'FL_UDSD'
) -> pd.DataFrame:
    """
    Combine multiple categories in the target column into single categories.

    Args:
        df (pd.DataFrame): Input dataframe
        combination_map (dict or None): Dictionary where keys are new category names and values
                                       are lists of categories to combine into that new category.
                                       If None, returns df unchanged.
                                       Example: {'SCD/Impaired': ['Subjective Cognitive Decline',
                                                                   'Impaired Not SCD/MCI']}
        target_col (str): Column containing categories to combine. Default: 'FL_UDSD'

    Returns:
        pd.DataFrame: DataFrame with combined categories
    """
    if combination_map is None:
        return df.copy()

    df = df.copy()

    # Apply each combination
    for new_category, old_categories in combination_map.items():
        df[target_col] = df[target_col].replace(old_categories, new_category)

    return df


def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Calculate evaluation metrics for a given set of true and predicted labels.

    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)

    Returns:
        dict: A dictionary containing accuracy, balanced accuracy, and f1 score.
              All values are rounded to 5 decimal places.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_score_macro': f1_score(y_true, y_pred, average='macro')
    }

    # Round to 5 decimal places
    metrics = {k: float(f'{v:.5f}') for k, v in metrics.items()}

    return metrics
