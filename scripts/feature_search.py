import cudf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score
from cuml.metrics import accuracy_score
import itertools
import time



def get_dataset_stats(df, train_df, test_df, classes, label_col):
    """
    Generate statistics about the dataset including sizes and class distributions.
    """
    full_dist = df[label_col].value_counts().sort_index()
    train_dist = train_df[label_col].value_counts().sort_index()
    test_dist = test_df[label_col].value_counts().sort_index()

    stats = {
        'Dataset': ['Full Dataset', 'Training Set', 'Test Set'],
        'Total Size': [len(df), len(train_df), len(test_df)]
    }

    for i, cls in enumerate(classes):
        stats[f'{cls}_Count'] = [
            full_dist.get(i, 0),
            train_dist.get(i, 0),
            test_dist.get(i, 0)
        ]

    stats_df = pd.DataFrame(stats)

    for i, cls in enumerate(classes):
        stats_df[f'{cls}_Percent'] = (stats_df[f'{cls}_Count'] / stats_df['Total Size'] * 100).round(2)

    return stats_df


def get_feature_info(train_cdf):
    """
    Generate information about features in the dataset.
    """
    exclude_cols = {'FL_UDSD_cat', 'PTID', 'FL_UDSD'}
    features = [col for col in train_cdf.columns if col not in exclude_cols]

    feature_info = []
    for feature in features:
        info = {
            'Feature_Name': feature,
            'Data_Type': str(train_cdf[feature].dtype),
            'Missing_Values': train_cdf[feature].isnull().sum(),
            'Unique_Values': train_cdf[feature].nunique(),
            'Min_Value': train_cdf[feature].min() if train_cdf[feature].dtype in ['int64', 'float64'] else 'N/A',
            'Max_Value': train_cdf[feature].max() if train_cdf[feature].dtype in ['int64', 'float64'] else 'N/A',
            'Mean_Value': train_cdf[feature].mean() if train_cdf[feature].dtype in ['int64', 'float64'] else 'N/A',
            'Std_Value': train_cdf[feature].std() if train_cdf[feature].dtype in ['int64', 'float64'] else 'N/A'
        }
        feature_info.append(info)

    return pd.DataFrame(feature_info)


def get_data_summary(df, train, test, classes, label_col):
    """
    Generate a concise summary of the dataset including total records,
    class distribution, and train/test split.
    """
    summary_info = []

    summary_info.append({
        'Metric': 'Total Records',
        'Value': len(df),
        'Description': 'Total number of records after data cleaning'
    })

    summary_info.append({
        'Metric': 'Training Records',
        'Value': f"{len(train)} ({len(train)/len(df)*100:.1f}%)",
        'Description': 'Number and percentage of records in training set'
    })

    summary_info.append({
        'Metric': 'Test Records',
        'Value': f"{len(test)} ({len(test)/len(df)*100:.1f}%)",
        'Description': 'Number and percentage of records in test set'
    })

    summary_info.append({
        'Metric': '--- CLASS DISTRIBUTION ---',
        'Value': '',
        'Description': ''
    })

    class_dist = df[label_col].value_counts().sort_index()
    for i, cls in enumerate(classes):
        count = class_dist.get(i, 0)
        percentage = count/len(df)*100 if len(df) > 0 else 0
        summary_info.append({
            'Metric': f'{cls}',
            'Value': f"{count} ({percentage:.1f}%)",
            'Description': f'Records in {cls} class'
        })

    return pd.DataFrame(summary_info)


def train_and_evaluate_model(feature_list, feature_name, train_cdf, test_cdf, label_col, classifier, classifier_name):
    """
    Train and evaluate a single classifier using specified features.

    Parameters:
    -----------
    feature_list : list
        List of feature names to use
    feature_name : str
        Name/description of the feature set
    train_cdf : cudf.DataFrame
        Training data
    test_cdf : cudf.DataFrame
        Test data
    label_col : str
        Name of the label column
    classifier : object
        Classifier instance (must have fit() and predict() methods)
    classifier_name : str
        Name of the classifier for reporting

    Returns:
    --------
    dict : Dictionary containing all evaluation metrics
    """
    # Prepare data
    X_train = train_cdf[feature_list]
    y_train = train_cdf[label_col]
    X_test = test_cdf[feature_list]
    y_test = test_cdf[label_col]

    # Train model
    classifier.fit(X_train, y_train)

    # Get predictions
    train_pred = classifier.predict(X_train)
    test_pred = classifier.predict(X_test)

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    train_balanced_accuracy = balanced_accuracy_score(y_train.to_numpy(), train_pred.to_numpy())
    test_balanced_accuracy = balanced_accuracy_score(y_test.to_numpy(), test_pred.to_numpy())
    f1_weighted = f1_score(y_test.to_numpy(), test_pred.to_numpy(), average='weighted')
    f1_macro = f1_score(y_test.to_numpy(), test_pred.to_numpy(), average='macro')

    return {
        'Feature_Set': feature_name,
        'Features': ', '.join(feature_list),
        'Num_Features': len(feature_list),
        'Train_Accuracy': train_accuracy,
        'Test_Accuracy': test_accuracy,
        'Train_Balanced_Accuracy': train_balanced_accuracy,
        'Test_Balanced_Accuracy': test_balanced_accuracy,
        'Overfitting_Gap_Regular': train_accuracy - test_accuracy,
        'Overfitting_Gap_Balanced': train_balanced_accuracy - test_balanced_accuracy,
        'F1_Weighted': f1_weighted,
        'F1_Macro': f1_macro
    }


def find_best_feature_combinations(train_cdf, test_cdf, label_col, classifier, classifier_name,
                                   num_features=3, top_n=5):
    """
    Finds the best feature combinations with exactly num_features.
    Ranking is done by BALANCED ACCURACY.

    Parameters:
    -----------
    train_cdf : cudf.DataFrame
        Training data
    test_cdf : cudf.DataFrame
        Test data
    label_col : str
        Name of the label column
    classifier : object
        Classifier instance (must have fit() and predict() methods)
    classifier_name : str
        Name of the classifier for reporting
    num_features : int
        Number of features in each combination
    top_n : int
        Number of top results to return

    Returns:
    --------
    dict : Dictionary containing top results and all results
    """
    # Get available features
    exclude_cols = {label_col, 'PTID', 'FL_UDSD'}
    all_features = [col for col in train_cdf.columns if col not in exclude_cols]

    combos = list(itertools.combinations(all_features, num_features))
    total_combos = len(combos)
    print(f"  Testing {num_features} features: {total_combos} combinations to try")

    combo_count = 0
    start_time = time.time()
    results = []

    for feature_combo in combos:
        combo_count += 1
        if combo_count % 100 == 0 or combo_count == total_combos:
            elapsed = time.time() - start_time
            print(f"    Progress: {combo_count}/{total_combos} combinations, elapsed: {elapsed:.1f}s")

        try:
            res = train_and_evaluate_model(
                feature_list=list(feature_combo),
                feature_name=','.join(feature_combo),
                train_cdf=train_cdf,
                test_cdf=test_cdf,
                label_col=label_col,
                classifier=classifier,
                classifier_name=classifier_name
            )
            results.append(res)
        except Exception as e:
            print(f"    Error with features {feature_combo}: {e}")

    # Convert to DataFrame and sort by balanced accuracy
    results_df = pd.DataFrame(results)
    top_results = results_df.sort_values('Test_Balanced_Accuracy', ascending=False).head(top_n)

    return {
        'top_results': top_results,
        'all_results': results_df
    }


def load_and_prepare_data(csv_path):
    """Load and prepare the data for feature search"""
    print("Loading and preparing data...")

    # Load data
    df = pd.read_csv(csv_path)
    print(f"Initial data shape: {df.shape}")

    # Select relevant columns
    df = df[['PTID', 'FL_MMSE','CDRSUM', 'CDRGLOB', 'HVLT_DR', 'COMBINED_NE4S', 'AMYLPET',
           'LASSI_A_CR2', 'LASSI_B_CR1', 'LASSI_B_CR2',
           'PTAU_217_CONCNTRTN',
           'FL_UDSD']]
    print(f"After column selection: {df.shape}")

    # Clean data
    df = df[df['FL_UDSD'] != 'Unknown']
    print(f"After removing 'Unknown' FL_UDSD: {df.shape}")

    df = df.dropna()
    print(f"After removing NaN values: {df.shape}")

    df = df[df['FL_MMSE'] != -1]
    print(f"After removing FL_MMSE == -1: {df.shape}")

    # Create categorical encoding
    cat_order = ['Normal cognition', 'Subjective Cognitive Decline', 'Impaired Not SCD/MCI',
                 'Early MCI', 'Late MCI', 'Dementia']
    df['FL_UDSD'] = pd.Categorical(df['FL_UDSD'], categories=cat_order, ordered=True)
    df['FL_UDSD_cat'] = df['FL_UDSD'].cat.codes

    # Print class distribution
    print("\nClass distribution in full dataset:")
    class_dist = df['FL_UDSD'].value_counts().sort_index()
    for cls, count in class_dist.items():
        print(f"  {cls}: {count} ({count/len(df)*100:.1f}%)")

    # Split data
    train, test = train_test_split(
        df,
        test_size=0.20,
        stratify=df['FL_UDSD_cat'],
        random_state=42
    )

    # Convert to cudf
    train_cdf = cudf.from_pandas(train)
    test_cdf = cudf.from_pandas(test)

    print(f"\nData prepared: {len(df)} total samples, {len(train)} train, {len(test)} test")

    return df, train, test, train_cdf, test_cdf, cat_order


def feature_search(df, train, test, train_cdf, test_cdf, classes, classifier, classifier_name,
                   label_col='FL_UDSD_cat', min_features=2, top_n=10, output_file=None):
    """
    Run comprehensive feature search from min_features to max available features.
    Save all results to an Excel file with multiple sheets.
    Rankings are done by BALANCED ACCURACY.

    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    train : pd.DataFrame
        Training data
    test : pd.DataFrame
        Test data
    train_cdf : cudf.DataFrame
        Training data in cuDF format
    test_cdf : cudf.DataFrame
        Test data in cuDF format
    classes : list
        List of class names
    classifier : object
        Classifier instance (must have fit() and predict() methods)
    classifier_name : str
        Name of the classifier for reporting
    label_col : str
        Name of the label column
    min_features : int
        Minimum number of features to test
    top_n : int
        Number of top results to save for each feature count
    output_file : str
        Path to save Excel results (if None, will auto-generate)

    Returns:
    --------
    tuple : (summary_df, all_results_dict, top_results_dict, dataset_stats, feature_info, data_summary)
    """
    # Determine available features
    exclude_cols = {label_col, 'PTID', 'FL_UDSD'}
    all_features = [col for col in train_cdf.columns if col not in exclude_cols]
    max_features = len(all_features)

    # Generate output filename if not provided
    if output_file is None:
        output_file = f"feature_search_{classifier_name.lower().replace(' ', '_')}_results.xlsx"

    print(f"\nFEATURE SEARCH - {classifier_name}")
    print("="*60)
    print(f"Available features: {all_features}")
    print(f"Running feature search from {min_features} to {max_features} features")
    print(f"Ranking method: BALANCED ACCURACY (better for imbalanced classes)")
    print(f"Results will be saved to: {output_file}")
    print("="*60)

    # Generate dataset statistics
    dataset_stats = get_dataset_stats(df, train, test, classes, label_col)
    feature_info = get_feature_info(train_cdf)
    data_summary = get_data_summary(df, train, test, classes, label_col)

    # Dictionary to store all results
    all_results_dict = {}
    top_results_dict = {}
    summary_data = []

    total_start_time = time.time()

    for num_features in range(min_features, max_features + 1):
        print(f"\n{'='*50}")
        print(f"PROCESSING {num_features} FEATURES")
        print(f"{'='*50}")

        feature_start_time = time.time()

        try:
            results = find_best_feature_combinations(
                train_cdf=train_cdf,
                test_cdf=test_cdf,
                label_col=label_col,
                classifier=classifier,
                classifier_name=classifier_name,
                num_features=num_features,
                top_n=top_n
            )

            # Store results
            all_results_dict[f'{num_features}_features'] = results['all_results']
            top_results_dict[f'{num_features}_features'] = results['top_results']

            # Calculate summary statistics
            all_res = results['all_results']
            best_score = all_res['Test_Balanced_Accuracy'].max()
            best_features = all_res.loc[all_res['Test_Balanced_Accuracy'].idxmax(), 'Features']

            feature_time = time.time() - feature_start_time

            summary_data.append({
                'Num_Features': num_features,
                'Total_Combinations': len(all_res),
                'Best_Balanced_Accuracy': best_score,
                'Best_Features': best_features,
                'Processing_Time_Minutes': feature_time / 60
            })

            print(f"  Completed in {feature_time:.1f} seconds")
            print(f"  Best balanced accuracy: {best_score:.4f}")
            print(f"  Best features: {best_features}")

        except Exception as e:
            print(f"  ERROR processing {num_features} features: {e}")
            continue

    total_time = time.time() - total_start_time
    print(f"\n{'='*50}")
    print(f"FEATURE SEARCH COMPLETED")
    print(f"Total processing time: {total_time/60:.1f} minutes")
    print(f"{'='*50}")

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Save all results to Excel
    print(f"\nSaving results to {output_file}...")

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Save dataset information
        data_summary.to_excel(writer, sheet_name='Data_Summary', index=False)
        dataset_stats.to_excel(writer, sheet_name='Dataset_Statistics', index=False)
        feature_info.to_excel(writer, sheet_name='Feature_Information', index=False)

        # Save summary
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Save all results for each feature count
        for sheet_name, df_results in all_results_dict.items():
            df_results.to_excel(writer, sheet_name=f'All_{sheet_name}', index=False)

        # Save top results for each feature count
        for sheet_name, df_results in top_results_dict.items():
            df_results.to_excel(writer, sheet_name=f'Top_{sheet_name}', index=False)

    print(f"Results successfully saved to {output_file}")

    # Print summaries
    print(f"\nDATA SUMMARY:")
    print(data_summary.to_string(index=False))

    print(f"\nDETAILED DATASET STATISTICS:")
    print(dataset_stats.to_string(index=False))

    print(f"\nSUMMARY OF RESULTS (Ranked by Balanced Accuracy):")
    print(summary_df.to_string(index=False))

    return summary_df, all_results_dict, top_results_dict, dataset_stats, feature_info, data_summary


def main():
    """Main function to run the feature search"""
    from cuml.ensemble import RandomForestClassifier
    from cuml.linear_model import LogisticRegression

    # Configuration
    csv_path = "/root/data/June/Data/Cardio_blood.csv"
    min_features = 2
    top_n = 10

    print("FEATURE SEARCH WITH CUSTOMIZABLE CLASSIFIER")
    print("="*60)

    # Load and prepare data
    df, train, test, train_cdf, test_cdf, classes = load_and_prepare_data(csv_path)

    # Example 1: Run with Random Forest
    rf_classifier = RandomForestClassifier(random_state=42)
    feature_search(
        df=df,
        train=train,
        test=test,
        train_cdf=train_cdf,
        test_cdf=test_cdf,
        classes=classes,
        classifier=rf_classifier,
        classifier_name="Random Forest",
        min_features=min_features,
        top_n=top_n,
        output_file="feature_search_random_forest_results.xlsx"
    )

    # Example 2: Run with Logistic Regression
    lr_classifier = LogisticRegression(max_iter=10000, random_state=42)
    feature_search(
        df=df,
        train=train,
        test=test,
        train_cdf=train_cdf,
        test_cdf=test_cdf,
        classes=classes,
        classifier=lr_classifier,
        classifier_name="Logistic Regression",
        min_features=min_features,
        top_n=top_n,
        output_file="feature_search_logistic_regression_results.xlsx"
    )

    print("\nSearch complete! Check the output files for detailed results.")
    print("\nEach Excel file contains:")
    print("- Data_Summary: Concise overview of total records, classes, and train/test split")
    print("- Dataset_Statistics: Detailed class distributions and split statistics")
    print("- Feature_Information: Details about each feature")
    print("- Summary: Best results for each feature count (ranked by balanced accuracy)")
    print("- All_X_features: Complete results for each feature count")
    print("- Top_X_features: Top results for each feature count (ranked by balanced accuracy)")


if __name__ == "__main__":
    main()
