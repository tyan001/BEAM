"""
Script to preprocess and save filtered data for ML model training.

This script loads raw data, applies preprocessing according to different
diagnosis grouping strategies, and saves the processed data to CSV files.

Usage:
    python preprocessing.py --input data/synthetic_data.csv --strategy all
    python preprocessing.py --input data/synthetic_data.csv --strategy original
    python preprocessing.py --input data/synthetic_data.cs --strategy scd_impaired
"""

import argparse
import pandas as pd
from pathlib import Path
from utils.data_preprocessing import (
    preprocess_data,
    get_diagnosis_config,
    combine_categories,
)


def create_processed_data(
    input_path: str, grouping_strategy: str, output_dir: str = "data/processed"
) -> str:
    """
    Process data according to grouping strategy and save to CSV.

    Args:
        input_path (str): Path to input CSV file
        grouping_strategy (str): One of 'original', 'scd_impaired', 'nc_impaired'
        output_dir (str): Directory to save processed files

    Returns:
        str: Path to saved file
    """
    # Load raw data
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows")

    # Get diagnosis configuration
    diagnosis_order, combination_map = get_diagnosis_config(grouping_strategy)
    print(f"\nGrouping strategy: {grouping_strategy}")
    print(f"Diagnosis order: {diagnosis_order}")

    # Combine categories if needed
    if combination_map:
        print(f"Combining categories: {combination_map}")
        df = combine_categories(df, combination_map)

    # Preprocess data
    print("\nPreprocessing data...")
    processed_df = preprocess_data(df, diagnosis_order=diagnosis_order)

    # Remove NaN values for complete cases
    processed_df = processed_df.dropna()
    print(f"After preprocessing and removing NaN: {len(processed_df)} rows")

    # Print class distribution
    print("\nClass distribution:")
    class_counts = processed_df["FL_UDSD"].value_counts().sort_index()
    for diagnosis, count in class_counts.items():
        print(f"  {diagnosis}: {count}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    input_filename = Path(input_path).stem
    output_file = output_path / f"{input_filename}_{grouping_strategy}.csv"

    # Save processed data
    processed_df.to_csv(output_file, index=False)
    print(f"\nSaved processed data to: {output_file}")
    print(f"Columns: {list(processed_df.columns)}")

    return str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess and save filtered data for ML training"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/synthetic_data.csv",
        help="Input CSV file path (default: data/synthetic_data.csv)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        choices=["all", "original", "scd_impaired", "nc_impaired"],
        help="Grouping strategy: 'all' processes all strategies, or choose specific one",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed files (default: data/processed)",
    )

    args = parser.parse_args()

    # Determine which strategies to process
    if args.strategy == "all":
        strategies = ["original", "scd_impaired", "nc_impaired"]
        print("Processing all grouping strategies...\n")
    else:
        strategies = [args.strategy]

    # Process each strategy
    saved_files = []
    for strategy in strategies:
        print("=" * 80)
        output_file = create_processed_data(
            input_path=args.input,
            grouping_strategy=strategy,
            output_dir=args.output_dir,
        )
        saved_files.append(output_file)
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Processed {len(strategies)} strategy/strategies:")
    for file in saved_files:
        print(f"  - {file}")
    print("\nProcessed data is ready for model training!")


if __name__ == "__main__":
    main()
