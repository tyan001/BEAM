import cudf
import pandas as pd
from sklearn.model_selection import train_test_split
from cuml.ensemble import RandomForestClassifier
from cuml.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt
from cuml.metrics import accuracy_score
import itertools
import time
import os
from pathlib import Path

def get_dataset_stats(df, train_df, test_df, classes, label_col):
    """
    Generate statistics about the dataset including sizes and class distributions.
    """
    # Get class distributions
    full_dist = df[label_col].value_counts().sort_index()
    train_dist = train_df[label_col].value_counts().sort_index()
    test_dist = test_df[label_col].value_counts().sort_index()

    # Initialize the stats dictionary
    stats = {
        'Dataset': ['Full Dataset', 'Training Set', 'Test Set'],
        'Total Size': [len(df), len(train_df), len(test_df)]
    }

    # Add class counts
    for i, cls in enumerate(classes):
        stats[f'{cls}_Count'] = [
            full_dist.get(i, 0),
            train_dist.get(i, 0),
            test_dist.get(i, 0)
        ]
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(stats)

    # Add percentage columns
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
    
    # Overall summary
    summary_info = []
    
    # Total records
    summary_info.append({
        'Metric': 'Total Records',
        'Value': len(df),
        'Description': 'Total number of records after data cleaning'
    })
    
    # Train/Test split
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
    
    # Add separator
    summary_info.append({
        'Metric': '--- CLASS DISTRIBUTION ---',
        'Value': '',
        'Description': ''
    })
    
    # Class distribution
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

def create_metrics_comparison(all_results_data):
    """
    Create a comprehensive metrics comparison showing both regular and balanced accuracy rankings.
    """
    comparison_data = []
    
    for feature_count, results_df in all_results_data.items():
        if len(results_df) == 0:
            continue
            
        # Get top performer by balanced accuracy (current ranking method)
        top_rf_balanced = results_df.loc[results_df['RandomForest_Test_Balanced_Accuracy'].idxmax()]
        top_lr_balanced = results_df.loc[results_df['LogisticRegression_Test_Balanced_Accuracy'].idxmax()]
        
        # Random Forest - Balanced Accuracy Ranking
        comparison_data.append({
            'Feature_Count': feature_count.replace('_features', ''),
            'Model': 'Random Forest',
            'Ranking_Method': 'Balanced Accuracy',
            'Best_Features': top_rf_balanced['Features'],
            'Train_Accuracy': f"{top_rf_balanced['RandomForest_Train_Accuracy']:.4f}",
            'Test_Accuracy': f"{top_rf_balanced['RandomForest_Test_Accuracy']:.4f}",
            'Train_Balanced_Accuracy': f"{top_rf_balanced['RandomForest_Train_Balanced_Accuracy']:.4f}",
            'Test_Balanced_Accuracy': f"{top_rf_balanced['RandomForest_Test_Balanced_Accuracy']:.4f}",
            'Overfitting_Gap_Regular': f"{top_rf_balanced['RandomForest_Train_Accuracy'] - top_rf_balanced['RandomForest_Test_Accuracy']:.4f}",
            'Overfitting_Gap_Balanced': f"{top_rf_balanced['RandomForest_Train_Balanced_Accuracy'] - top_rf_balanced['RandomForest_Test_Balanced_Accuracy']:.4f}",
            'f1_Weighted': f"{top_rf_balanced['RandomForest_F1_Weighted']:.4f}",
            'f1_Macro': f"{top_rf_balanced['RandomForest_F1_Macro']:.4f}"
        })
        
        
        # Logistic Regression - Balanced Accuracy Ranking
        comparison_data.append({
            'Feature_Count': feature_count.replace('_features', ''),
            'Model': 'Logistic Regression',
            'Ranking_Method': 'Balanced Accuracy',
            'Best_Features': top_lr_balanced['Features'],
            'Train_Accuracy': f"{top_lr_balanced['LogisticRegression_Train_Accuracy']:.4f}",
            'Test_Accuracy': f"{top_lr_balanced['LogisticRegression_Test_Accuracy']:.4f}",
            'Train_Balanced_Accuracy': f"{top_lr_balanced['LogisticRegression_Train_Balanced_Accuracy']:.4f}",
            'Test_Balanced_Accuracy': f"{top_lr_balanced['LogisticRegression_Test_Balanced_Accuracy']:.4f}",
            'Overfitting_Gap_Regular': f"{top_lr_balanced['LogisticRegression_Train_Accuracy'] - top_lr_balanced['LogisticRegression_Test_Accuracy']:.4f}",
            'Overfitting_Gap_Balanced': f"{top_lr_balanced['LogisticRegression_Train_Balanced_Accuracy'] - top_lr_balanced['LogisticRegression_Test_Balanced_Accuracy']:.4f}",
            'f1_Weighted': f"{top_lr_balanced['LogisticRegression_F1_Weighted']:.4f}",
            'f1_Macro': f"{top_lr_balanced['LogisticRegression_F1_Macro']:.4f}"
        })
        
    
    return pd.DataFrame(comparison_data)

def train_and_evaluate_models(feature_list, feature_name, train_cdf, test_cdf, label_col):
    """
    Train and evaluate both Random Forest and Logistic Regression models using specified features.
    """
    # Prepare data
    X_train = train_cdf[feature_list]
    y_train = train_cdf[label_col]
    X_test = test_cdf[feature_list]
    y_test = test_cdf[label_col]
    
    # Train and evaluate Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_balanced_accuracy = balanced_accuracy_score(y_test.to_numpy(), rf_pred.to_numpy())
    # Training accuracy for Random Forest
    rf_train_pred = rf.predict(X_train)
    rf_train_accuracy = accuracy_score(y_train, rf_train_pred)
    rf_train_balanced_accuracy = balanced_accuracy_score(y_train.to_numpy(), rf_train_pred.to_numpy())
    rf_f1_weighted = f1_score(y_test.to_numpy(), rf_pred.to_numpy(), average='weighted')
    rf_f1_macro = f1_score(y_test.to_numpy(), rf_pred.to_numpy(), average='macro')
    
    # Train and evaluate Logistic Regression
    logreg = LogisticRegression(max_iter=10000)
    logreg.fit(X_train, y_train)
    logreg_pred = logreg.predict(X_test)
    logreg_accuracy = accuracy_score(y_test, logreg_pred)
    logreg_balanced_accuracy = balanced_accuracy_score(y_test.to_numpy(), logreg_pred.to_numpy())
    # Training accuracy for Logistic Regression
    logreg_train_pred = logreg.predict(X_train)
    logreg_train_accuracy = accuracy_score(y_train, logreg_train_pred)
    logreg_train_balanced_accuracy = balanced_accuracy_score(y_train.to_numpy(), logreg_train_pred.to_numpy())
    logreg_f1_weighted = f1_score(y_test.to_numpy(), logreg_pred.to_numpy(), average='weighted')
    logreg_f1_macro = f1_score(y_test.to_numpy(), logreg_pred.to_numpy(), average='macro')
    
    return {
        'Feature_Set': feature_name,
        'Features': ', '.join(feature_list),
        'Num_Features': len(feature_list),
        'RandomForest_Train_Accuracy': rf_train_accuracy,
        'RandomForest_Train_Balanced_Accuracy': rf_train_balanced_accuracy,
        'RandomForest_Test_Accuracy': rf_accuracy,
        'RandomForest_Test_Balanced_Accuracy': rf_balanced_accuracy,
        'RandomForest_F1_Weighted': rf_f1_weighted,
        'RandomForest_F1_Macro': rf_f1_macro,
        'LogisticRegression_Train_Accuracy': logreg_train_accuracy,
        'LogisticRegression_Train_Balanced_Accuracy': logreg_train_balanced_accuracy,
        'LogisticRegression_Test_Accuracy': logreg_accuracy,
        'LogisticRegression_Test_Balanced_Accuracy': logreg_balanced_accuracy,
        'LogisticRegression_F1_Weighted': logreg_f1_weighted,
        'LogisticRegression_F1_Macro': logreg_f1_macro
    }

def find_best_feature_combinations(train_cdf, test_cdf, label_col, num_features=3, top_n=5):
    """
    Finds the best feature combinations with exactly num_features for both RandomForest and LogisticRegression.
    Ranking is done by BALANCED ACCURACY.
    """

    # Use all columns except PTID, label_col, and FL_UDSD
    exclude_cols = {label_col, 'PTID', 'FL_UDSD'}
    all_features = [col for col in train_cdf.columns if col not in exclude_cols]

    k = num_features
    combos = list(itertools.combinations(all_features, k))
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
            res = train_and_evaluate_models(
                list(feature_combo),
                feature_name=','.join(feature_combo),
                train_cdf=train_cdf,
                test_cdf=test_cdf,
                label_col=label_col
            )
            results.append(res)
        except Exception as e:
            print(f"    Error with features {feature_combo}: {e}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    # Get top N for each model - RANKED BY BALANCED ACCURACY
    top_rf = results_df.sort_values('RandomForest_Test_Balanced_Accuracy', ascending=False).head(top_n)
    top_lr = results_df.sort_values('LogisticRegression_Test_Balanced_Accuracy', ascending=False).head(top_n)
    
    return {
        'top_random_forest': top_rf,
        'top_logistic_regression': top_lr,
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
    
    # combined the Normal cognition and Subjective Cognitive Decline and call it Normal cognition and SCD
    df['FL_UDSD'] = df['FL_UDSD'].replace('Normal cognition', 'Normal cognition and SCD')
    df['FL_UDSD'] = df['FL_UDSD'].replace('Subjective Cognitive Decline', 'Normal cognition and SCD')
    
    # Create categorical encoding
    cat_order = ['Normal cognition and SCD','Impaired Not SCD/MCI', 'Early MCI', 'Late MCI', 'Dementia']
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

def comprehensive_feature_search(df, train, test, train_cdf, test_cdf, classes, label_col='FL_UDSD_cat', min_features=2, top_n=10, output_file='comprehensive_feature_search_results.xlsx'):
    """
    Run comprehensive feature search from min_features to max available features.
    Save all results to an Excel file with multiple sheets.
    Rankings are done by BALANCED ACCURACY.
    """
    
    # Determine available features
    exclude_cols = {label_col, 'PTID', 'FL_UDSD'}
    all_features = [col for col in train_cdf.columns if col not in exclude_cols]
    max_features = len(all_features)
    
    print(f"Available features: {all_features}")
    print(f"Running feature search from {min_features} to {max_features} features")
    print(f"Rankings are based on BALANCED ACCURACY (better for imbalanced classes)")
    print(f"Results will be saved to: {output_file}")
    
    # Generate dataset statistics
    dataset_stats = get_dataset_stats(df, train, test, classes, label_col)
    feature_info = get_feature_info(train_cdf)
    data_summary = get_data_summary(df, train, test, classes, label_col)
    
    # Dictionary to store all results
    all_results_data = {}
    top_rf_data = {}
    top_lr_data = {}
    
    # Summary data for overview
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
                num_features=num_features,
                top_n=top_n
            )
            
            # Store results
            all_results_data[f'{num_features}_features'] = results['all_results']
            top_rf_data[f'{num_features}_features_RF'] = results['top_random_forest']
            top_lr_data[f'{num_features}_features_LR'] = results['top_logistic_regression']
            
            # Calculate summary statistics
            all_res = results['all_results']
            best_rf_score = all_res['RandomForest_Test_Balanced_Accuracy'].max()
            best_lr_score = all_res['LogisticRegression_Test_Balanced_Accuracy'].max()
            
            # Get best feature sets
            best_rf_features = all_res.loc[all_res['RandomForest_Test_Balanced_Accuracy'].idxmax(), 'Features']
            best_lr_features = all_res.loc[all_res['LogisticRegression_Test_Balanced_Accuracy'].idxmax(), 'Features']
            
            feature_time = time.time() - feature_start_time
            
            summary_data.append({
                'Num_Features': num_features,
                'Total_Combinations': len(all_res),
                'Best_RF_Balanced_Score': best_rf_score,
                'Best_RF_Features': best_rf_features,
                'Best_LR_Balanced_Score': best_lr_score,
                'Best_LR_Features': best_lr_features,
                'Processing_Time_Minutes': feature_time / 60
            })
            
            print(f"  Completed in {feature_time:.1f} seconds")
            print(f"  Best RF balanced score: {best_rf_score:.4f}")
            print(f"  Best LR balanced score: {best_lr_score:.4f}")
            
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
    
    # Create comprehensive metrics comparison
    metrics_comparison = create_metrics_comparison(all_results_data)
    
    # Save all results to Excel
    print(f"\nSaving results to {output_file}...")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Save dataset statistics and feature info first
        data_summary.to_excel(writer, sheet_name='Data_Summary', index=False)
        dataset_stats.to_excel(writer, sheet_name='Dataset_Statistics', index=False)
        feature_info.to_excel(writer, sheet_name='Feature_Information', index=False)
        
        # Save summary (ranked by balanced accuracy)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Save comprehensive metrics comparison
        metrics_comparison.to_excel(writer, sheet_name='Metrics_Comparison', index=False)
        
        # Save all results for each feature count
        for sheet_name, df_results in all_results_data.items():
            df_results.to_excel(writer, sheet_name=f'All_{sheet_name}', index=False)
        
        # Save top Random Forest results (ranked by balanced accuracy)
        for sheet_name, df_results in top_rf_data.items():
            df_results.to_excel(writer, sheet_name=f'Top_{sheet_name}', index=False)
        
        # Save top Logistic Regression results (ranked by balanced accuracy)
        for sheet_name, df_results in top_lr_data.items():
            df_results.to_excel(writer, sheet_name=f'Top_{sheet_name}', index=False)
    
    print(f"Results successfully saved to {output_file}")
    
    # Print data summary
    print(f"\nDATA SUMMARY:")
    print(data_summary.to_string(index=False))
    
    # Print dataset statistics
    print(f"\nDETAILED DATASET STATISTICS:")
    print(dataset_stats.to_string(index=False))
    
    # Print summary (ranked by balanced accuracy)
    print(f"\nSUMMARY OF RESULTS (Ranked by Balanced Accuracy):")
    print(summary_df.to_string(index=False))
    
    return summary_df, all_results_data, top_rf_data, top_lr_data, dataset_stats, feature_info, data_summary, metrics_comparison

def main():
    """Main function to run the comprehensive feature search"""
    
    # Configuration
    csv_path = "/root/data/June/Data/Cardio_blood.csv"
    output_file = "comprehensive_feature_search_results_NC_SCD.xlsx"
    min_features = 2
    top_n = 10  # Number of top results to save for each feature count
    
    print("COMPREHENSIVE FEATURE SEARCH WITH DATASET STATISTICS")
    print("="*60)
    print("RANKING METHOD: BALANCED ACCURACY (better for imbalanced classes)")
    print("="*60)
    
    # Load and prepare data
    df, train, test, train_cdf, test_cdf, classes = load_and_prepare_data(csv_path)
    
    # Run comprehensive search
    summary_df, all_results, top_rf, top_lr, dataset_stats, feature_info, data_summary, metrics_comparison = comprehensive_feature_search(
        df=df,
        train=train,
        test=test,
        train_cdf=train_cdf,
        test_cdf=test_cdf,
        classes=classes,
        min_features=min_features,
        top_n=top_n,
        output_file=output_file
    )
    
    print(f"\nSearch complete! Check {output_file} for detailed results.")
    print("\nExcel file contains:")
    print("- Data_Summary: Concise overview of total records, classes, and train/test split")
    print("- Dataset_Statistics: Detailed class distributions and split statistics")
    print("- Feature_Information: Details about each feature")
    print("- Summary: Best results for each feature count (ranked by balanced accuracy)")
    print("- Metrics_Comparison: Comprehensive comparison of all metrics (training, testing, regular & balanced accuracy)")
    print("- All_X_features: Complete results for each feature count")
    print("- Top_X_features_RF/LR: Top 10 results for each model and feature count (ranked by balanced accuracy)")

if __name__ == "__main__":
    main() 