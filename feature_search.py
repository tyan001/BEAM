import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from itertools import combinations
from tqdm import tqdm
from pathlib import Path

def preprocess_data(df: pd.DataFrame, target_col: str='FL_UDSD', diagnosis_order: list=None) -> pd.DataFrame:
    """
    Preprocess the data by splitting into train and test sets.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        random_state (int): Random state for reproducibility.
        target_col (str): The target column for stratification.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The train and test dataframes.
    """
    
    # Clean data
    filter_df = df[df[target_col] != 'Unknown'] # Remove rows with unknown target values
    filter_df = filter_df[filter_df["MMSE"] != -1] # Remove rows with invalid MMSE values
    
    # Convert columns to categorical if needed
    filter_df['APOE'] = filter_df['APOE'].astype('category')
    filter_df['AMYLPET'] = filter_df['AMYLPET'].astype('category')
    
    # Encode the target variable as an ordered categorical variable.    
    filter_df['FL_UDSD'] = pd.Categorical(filter_df['FL_UDSD'], categories=diagnosis_order, ordered=True)
    filter_df['FL_UDSD_cat'] = filter_df['FL_UDSD'].cat.codes 
    
    
    return filter_df

def calculate_metrics(y_true, y_pred) -> dict:
    """
    Calculate evaluation metrics for a given set of true and predicted labels.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        mertrics: A dictionary containing accuracy, balanced accuracy, and f1 score.
    """
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_score_macro': f1_score(y_true, y_pred, average='macro')
    }
    
    metrics = {k: float(f'{v:.5f}') for k, v in metrics.items()}
    
    return metrics


def train_and_evaluate_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model,
    target_col: str = 'FL_UDSD_cat'
) -> dict:
    """
    Train and evaluate a single model.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        model: An instance of a sklearn model to train and evaluate
        target_col: Name of target column to exclude from features (default: 'FL_UDSD_cat')
    
    returns:
        dict: A dictionary containing training and testing metrics (accuracy, balanced accuracy, f1 score) for both train and test sets.
    
    """
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_metrics = calculate_metrics(y_train, train_preds)
    test_metrics = calculate_metrics(y_test, test_preds)
    
    metrics = {
        'train_accuracy': train_metrics['accuracy'],
        'test_accuracy': test_metrics['accuracy'],
        'train_balanced_accuracy': train_metrics['balanced_accuracy'],
        'test_balanced_accuracy': test_metrics['balanced_accuracy'],
        'train_f1_macro_score': train_metrics['f1_score_macro'],
        'test_f1_macro_score': test_metrics['f1_score_macro']
    }
    
    return metrics


def comprehensive_feature_search(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = 'FL_UDSD_cat',
    min_features: int = 2,
    max_features: int = None,
    models: dict = None
):
    """
    Test all possible feature subsets with multiple models.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        target_col: Name of target column to exclude from features
        min_features: Minimum number of features in a subset (default: 2)
        max_features: Maximum number of features in a subset (default: all features)
        models: Dictionary of model names and their corresponding sklearn model instances
            ie. {'RandomForest': RandomForestClassifier(random_state=42), 'LogisticRegression': LogisticRegression(max_iter=10000, random_state=42)}
    
    Returns:
        pd.DataFrame: Results sorted by test balanced accuracy
    """
    
    # Get all available features (excluding target column)
    all_features = [col for col in train_df.columns if col != target_col]
    
    if max_features is None:
        max_features = len(all_features)
    
    print(f"Total features available: {len(all_features)}")
    print(f"Features: {all_features}\n")
    
    # Define models to test
    if models is None:
        models = {
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
            'LogisticRegression': LogisticRegression(max_iter=10000, random_state=42)
        }
    
    results = []
    
    # Try all subset sizes from min_features to max_features
    for n_features in range(min_features, max_features + 1):
        # Generate all combinations of n_features
        feature_combinations = list(combinations(all_features, n_features))
        
        print(f"Testing {len(feature_combinations)} combinations with {n_features} features...")
        
        # Test each combination with each model
        for features in tqdm(feature_combinations, desc=f"{n_features} features"):
            features = list(features)
            
            # Prepare data with selected features
            X_train = train_df[features]
            y_train = train_df[target_col]
            X_test = test_df[features]
            y_test = test_df[target_col]
            
            # Test each model
            for model_name, model in models.items():
                try:
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    train_preds = model.predict(X_train)
                    test_preds = model.predict(X_test)
                    
                    train_metrics = calculate_metrics(y_train, train_preds)
                    test_metrics = calculate_metrics(y_test, test_preds)
                    
                    # Calculate metrics
                    result = {
                        'model': model_name,
                        'n_features': n_features,
                        'features': ', '.join(features),
                        'train_accuracy': train_metrics['accuracy'],
                        'test_accuracy': test_metrics['accuracy'],
                        'train_balanced_acc': train_metrics['balanced_accuracy'],
                        'test_balanced_acc': test_metrics['balanced_accuracy'],
                        'train_f1_macro_score': train_metrics['f1_score_macro'],
                        'test_f1_macro_score': test_metrics['f1_score_macro']
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    # Skip combinations that cause errors
                    print(f"Error with {model_name} and features {features}: {e}")
                    continue
    
    # Convert to DataFrame and sort by test balanced accuracy
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_balanced_acc', ascending=False)
    
    return results_df


def combine_categories(df: pd.DataFrame, combination_map: dict, target_col: str = 'FL_UDSD') -> pd.DataFrame:
    """
    Combine multiple categories in the target column into single categories.
    
    Args:
        df: Input dataframe
        target_col: Column containing categories to combine
        combination_map: Dictionary where keys are new category names and values are lists of 
                        categories to combine into that new category.
                        Example: {'SCD/Impaired': ['Subjective Cognitive Decline', 'Impaired Not SCD/MCI'],
                                 'Normal/SCD': ['Normal cognition', 'Subjective Cognitive Decline']}
    
    Returns:
        pd.DataFrame: DataFrame with combined categories
    """
    df = df.copy()
    
    # Apply each combination
    for new_category, old_categories in combination_map.items():
        df[target_col] = df[target_col].replace(old_categories, new_category)
    
    return df


def save_results(
    results_df: pd.DataFrame,
    output_dir: str,
    sort_by: str = 'test_f1_macro',
    ascending: bool = False
):
    """
    Save feature search results to separate Excel files for each model.
    Each Excel file contains sheets organized by feature count.
    
    Args:
        results_df: DataFrame containing feature search results from comprehensive_feature_search
        output_dir: Directory where the Excel files will be saved
        sort_by: Column name to sort results by (default: 'test_balanced_acc')
        ascending: Sort order (default: False for descending)
    
    Returns:
        None. Saves Excel files to disk.
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get unique models
    models = results_df['model'].unique()
    
    # Create a separate Excel file for each model
    for model in models:
        # Filter data for this model
        model_df = results_df[results_df['model'] == model].copy()
        
        # Create file path
        file_path = Path(output_dir) / f"{model}.xlsx"
        
        # Create Excel writer
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Get unique feature counts for this model
            feature_counts = sorted(model_df['n_features'].unique())
            
            # Create a sheet for each feature count
            for n_features in feature_counts:
                # Filter data for this feature count
                filtered_df = model_df[model_df['n_features'] == n_features].copy()
                
                if not filtered_df.empty:
                    # Sort the filtered data
                    filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)
                    
                    # Create sheet name
                    sheet_name = f"{n_features}_features"
                    
                    # Write to Excel
                    filtered_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Also create a summary sheet with all results for this model
            model_df_sorted = model_df.sort_values(sort_by, ascending=ascending)
            model_df_sorted.to_excel(writer, sheet_name='All_Results', index=False)
        
        print(f"Created {file_path} with {len(feature_counts)} feature count sheets")
    
    print(f"\nSaved {len(models)} Excel files to {output_dir}")
    print(f"Models: {', '.join(models)}")


def get_diagnosis_config(grouping_strategy: str = "original"):
    """
    Get diagnosis order and combination map based on grouping strategy.
    
    Parameters:
    -----------
    grouping_strategy : str
        "original" - Original 6 categories (no combination)
        "scd_impaired" - Combine SCD and Impaired Not SCD/MCI
        "nc_impaired" - Combine Normal Cognition and Impaired Not SCD/MCI
    
    Returns:
    --------
    tuple: (diagnosis_order, combination_map)
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
        raise ValueError(f"Invalid grouping strategy: '{grouping_strategy}'. Must be one of: {list(configs.keys())}")
    
    config = configs[grouping_strategy]
    return config['diagnosis_order'], config['combination_map']
    

if __name__ == "__main__":
    
    # Load data
    df = pd.read_csv('data/synthetic_data.csv')
    
    group_strategy = "original"  # Change this to "scd_impaired" or "nc_impaired" as needed
    
    diagnosis_order, combination_map = get_diagnosis_config(grouping_strategy="original")
    
    # # Preprocess data
    # diagnosis_order = ['Normal cognition', 'Subjective Cognitive Decline', 'Impaired Not SCD/MCI',
    #                    'Early MCI', 'Late MCI', 'Dementia']
    
    # # Combine categories as needed
    # # For example, if you want to combine 'Subjective Cognitive Decline' and 'Impaired Not SCD/MCI' into a single category called 'SCD/Impaired', you can do that as follows:
    # # combination_map = {
    # # 'SCD/Impaired': ['Subjective Cognitive Decline', 'Impaired Not SCD/MCI'],
    # # }
    
    # # diagnosis_order = [
    # #     'Normal cognition', 
    # #     'SCD/Impaired',
    # #     'Early MCI', 
    # #     'Late MCI', 
    # #     'Dementia'
    # # ]
    
    # # df = combine_categories(df, combination_map, target_col='FL_UDSD')
    # ##################################################################### 
    # combination_map = {
    # 'NC/Impaired': ['Normal cognition', 'Impaired Not SCD/MCI'],
    # }
    
    # diagnosis_order = [
    #     'NC/Impaired', 
    #     'Subjective Cognitive Decline',
    #     'Early MCI', 
    #     'Late MCI', 
    #     'Dementia'
    # ]
    
    # df = combine_categories(df, combination_map, target_col='FL_UDSD')
    # #####################################################################
    
    processed_df = preprocess_data(df, target_col='FL_UDSD', diagnosis_order=diagnosis_order)
    processed_df.dropna(inplace=True) # Drop rows with missing values after preprocessing. Only complete cases will be used for feature search.
    
    # save the data statistics on classes and how many samples are in each class to a csv file
    class_counts = processed_df['FL_UDSD'].value_counts().reset_index()
    class_counts.columns = ['FL_UDSD', 'count']
    
    results_path = Path('results')
    results_path.mkdir(parents=True, exist_ok=True)
    class_counts.to_csv('results/class_counts.csv', index=False)
    
    # No longer need this column for modeling
    processed_df.drop(columns=['FL_UDSD'], inplace=True)
    
    # Split into train and test sets
    train_df, test_df = train_test_split(processed_df, test_size=0.2, random_state=42, stratify=processed_df['FL_UDSD_cat'])
    
    max_features = len(train_df.columns) - 1  # Exclude target column
    # Run comprehensive feature search
    results_df = comprehensive_feature_search(train_df=train_df, test_df=test_df, target_col='FL_UDSD_cat', min_features=2, max_features=max_features)
    
    # Save results to CSV
    #results_df.to_csv('../results/feature_search_results.csv', index=False)
    
    # save results to Excel files by model. sort by test_f1_macro_score in descending order.
    save_results(results_df, 'results/',sort_by='test_f1_macro_score', ascending=False)