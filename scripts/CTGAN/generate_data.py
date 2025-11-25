import pickle
import argparse
import pandas as pd
from sdv.sampling import Condition
import numpy as np
import joblib

def round_to_half(value):
    """Round a value to the nearest 0.5 increment."""
    return np.round(value * 2) / 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic data using a trained CTGAN model')
    parser.add_argument('--model', type=str, default='ctgan_model_baseline.joblib',
                        help='Path to the pickled CTGAN model file')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Number of synthetic samples to generate')
    parser.add_argument('--output', type=str, default='synthetic_data.csv',
                        help='Output CSV file name')
    parser.add_argument('--balance', action='store_true',
                        help='Generate balanced dataset with equal samples per class')
    parser.add_argument('--target_column', type=str, default='FL_UDSD',
                        help='Target column name for class balancing (used with --balance)')

    args = parser.parse_args()

    with open(args.model, 'rb') as f:
        model = joblib.load(f)

    if args.balance:
        # Generate balanced synthetic data
        target_column = args.target_column
        print(f"Generating balanced data for column: {target_column}")

        # Sample a small batch to identify unique classes
        sample_batch = model.sample(1000)
        unique_classes = sample_batch[target_column].unique()
        n_classes = len(unique_classes)

        print(f"Found {n_classes} unique classes: {unique_classes}")

        # Calculate samples per class
        samples_per_class = args.n_samples // n_classes
        remainder = args.n_samples % n_classes

        print(f"Generating {samples_per_class} samples per class (total: {samples_per_class * n_classes})")

        # Generate balanced synthetic data
        synthetic_data_list = []
        for i, class_value in enumerate(unique_classes):
            # Add extra sample to first classes if there's a remainder
            n_samples_for_class = samples_per_class + (1 if i < remainder else 0)

            # Create Condition object for this class
            condition = Condition(
                num_rows=n_samples_for_class,
                column_values={target_column: class_value}
            )

            # Sample with conditions
            class_data = model.sample_from_conditions(conditions=[condition])
            synthetic_data_list.append(class_data)
            print(f"Generated {len(class_data)} samples for class '{class_value}'")

        # Combine all class data
        synthetic_data = pd.concat(synthetic_data_list, ignore_index=True)

        # Shuffle the dataset
        synthetic_data = synthetic_data.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        # Original behavior: generate unbalanced data
        synthetic_data = model.sample(args.n_samples)
        

    synthetic_data.rename(columns={
        'FL_MMSE': 'MMSE',
        'COMBINED_NE4S': "APOE"
    }, inplace=True)
    synthetic_data['CDRSUM'] = synthetic_data['CDRSUM'].apply(round_to_half)
    synthetic_data.to_csv(args.output, index=False)

    # Print summary
    if args.balance:
        target_column_renamed = 'MMSE' if args.target_column == 'FL_MMSE' else args.target_column
        if target_column_renamed in synthetic_data.columns:
            print(f"\nClass distribution in generated data:")
            print(synthetic_data[target_column_renamed].value_counts().sort_index())

    print(f"\nGenerated {len(synthetic_data)} synthetic samples and saved to {args.output}")