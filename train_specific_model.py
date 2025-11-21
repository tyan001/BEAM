import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from pathlib import Path

# Define the specific features to use
SELECTED_FEATURES = [
    'CDRSUM',
    'CDRGLOB',
    'AMYLPET',
    'APOE',
    'PTAU_217_CONCNTRTN',
    'HVLT_DR',
    'LASSI_B_CR1',
    'LASSI_B_CR2'
]

def preprocess_data(df, target_col='FL_UDSD', diagnosis_order=None):
    """
    Preprocess the data and prepare it for modeling.
    """
    # Clean data
    filter_df = df[df[target_col] != 'Unknown'].copy()
    filter_df = filter_df[filter_df["MMSE"] != -1]

    # Convert columns to categorical
    filter_df['APOE'] = filter_df['APOE'].astype('category')
    filter_df['AMYLPET'] = filter_df['AMYLPET'].astype('category')

    # Encode the target variable
    filter_df['FL_UDSD'] = pd.Categorical(filter_df['FL_UDSD'], categories=diagnosis_order, ordered=True)
    filter_df['FL_UDSD_cat'] = filter_df['FL_UDSD'].cat.codes

    return filter_df

def train_model(features_list, output_dir='models'):
    """
    Train a Random Forest model using specific features.

    Args:
        features_list: List of feature names to use
        output_dir: Directory to save the trained model
    """
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/cardio_blood.csv')

    # Diagnosis order
    diagnosis_order = [
        'Normal cognition',
        'Subjective Cognitive Decline',
        'Impaired Not SCD/MCI',
        'Early MCI',
        'Late MCI',
        'Dementia'
    ]

    # Preprocess
    print("Preprocessing data...")
    processed_df = preprocess_data(df, target_col='FL_UDSD', diagnosis_order=diagnosis_order)

    # Select only the features we want plus the target
    required_cols = features_list + ['FL_UDSD_cat', 'FL_UDSD']
    processed_df = processed_df[required_cols]

    # Drop rows with NaN values
    print(f"Shape before dropping NaN: {processed_df.shape}")
    processed_df = processed_df.dropna()
    print(f"Shape after dropping NaN: {processed_df.shape}")

    # Separate features and target
    X = processed_df[features_list]
    y = processed_df['FL_UDSD_cat']

    print(f"\nFeatures used: {features_list}")
    print(f"Number of features: {len(features_list)}")
    print(f"Total samples: {len(X)}")
    print(f"\nClass distribution:")
    class_counts = processed_df['FL_UDSD'].value_counts()
    print(class_counts)

    # Split data
    print("\nSplitting data into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Also get the labels for the test set
    y_test_labels = processed_df.loc[y_test.index, 'FL_UDSD']

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Make predictions
    print("\nEvaluating model...")
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Calculate metrics
    train_metrics = {
        'accuracy': accuracy_score(y_train, train_preds),
        'balanced_accuracy': balanced_accuracy_score(y_train, train_preds),
        'f1_macro': f1_score(y_train, train_preds, average='macro')
    }

    test_metrics = {
        'accuracy': accuracy_score(y_test, test_preds),
        'balanced_accuracy': balanced_accuracy_score(y_test, test_preds),
        'f1_macro': f1_score(y_test, test_preds, average='macro')
    }

    # Print results
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"Train Accuracy: {train_metrics['accuracy']:.5f}")
    print(f"Train Balanced Accuracy: {train_metrics['balanced_accuracy']:.5f}")
    print(f"Train F1 Macro Score: {train_metrics['f1_macro']:.5f}")
    print()
    print(f"Test Accuracy: {test_metrics['accuracy']:.5f}")
    print(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.5f}")
    print(f"Test F1 Macro Score: {test_metrics['f1_macro']:.5f}")
    print("="*60)

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_preds, target_names=diagnosis_order))

    print("\nConfusion Matrix (Test Set):")
    print(confusion_matrix(y_test, test_preds))

    # Feature importance
    print("\nFeature Importances:")
    feature_importance = pd.DataFrame({
        'feature': features_list,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.to_string(index=False))

    # Save model
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(output_dir) / 'random_forest_8features.pkl'

    model_info = {
        'model': model,
        'features': features_list,
        'diagnosis_order': diagnosis_order,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_importance': feature_importance
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_info, f)

    print(f"\nModel saved to: {model_path}")

    # Save feature importance to CSV
    importance_path = Path(output_dir) / 'feature_importance.csv'
    feature_importance.to_csv(importance_path, index=False)
    print(f"Feature importance saved to: {importance_path}")

    # Save test dataset with labels and predictions
    test_dataset = X_test.copy()
    test_dataset['FL_UDSD_cat'] = y_test
    test_dataset['FL_UDSD'] = y_test_labels.values
    test_dataset['predicted_cat'] = test_preds
    test_dataset['predicted_label'] = [diagnosis_order[pred] for pred in test_preds]
    test_dataset['correct_prediction'] = (y_test.values == test_preds)

    test_dataset_path = Path(output_dir) / 'test_dataset.csv'
    test_dataset.to_csv(test_dataset_path, index=False)
    print(f"Test dataset saved to: {test_dataset_path}")
    print(f"  - Contains {len(test_dataset)} samples")
    print(f"  - Includes features, true labels, predictions, and correctness")

    return model_info

if __name__ == "__main__":
    print("Training Random Forest with 8 specific features")
    print("="*60)

    model_info = train_model(SELECTED_FEATURES)

    print("\nTraining complete!")
