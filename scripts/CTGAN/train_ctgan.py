import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
from sdv.evaluation.single_table import evaluate_quality
from typing import List, Dict, Optional, Tuple
import joblib
from rdt.transformers import FloatFormatter



def round_to_half(value):
    """Round a value to the nearest 0.5 increment."""
    return np.round(value * 2) / 2


def train_synthesizer(df: pd.DataFrame, metadata: Metadata, config: Dict, enable_gpu: bool = False) -> CTGANSynthesizer:
    """
        Function to train a CTGAN synthesizer on the given dataframe and metadata using the specified configuration.

        Args:
            df: The input dataframe to train the synthesizer on.
            metadata: The metadata object describing the structure of the data.
            config: A dictionary containing the configuration parameters for the synthesizer.
    """

    synthesizer = CTGANSynthesizer(
        metadata=metadata,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        embedding_dim=config['embedding_dim'],
        generator_dim=config['generator_dim'],
        discriminator_dim=config['discriminator_dim'],
        generator_lr=config['generator_lr'],
        discriminator_lr=config['discriminator_lr'],
        cuda=enable_gpu,
        # verbose=True
    )
    synthesizer.fit(df)
    return synthesizer


if __name__ == "__main__":

    df = pd.read_csv("../../data/Cardio_blood.csv")
    feature_cols = ['FL_MMSE', 'CDRSUM', 'CDRGLOB', 'HVLT_DR', 'LASSI_A_CR2', 'LASSI_B_CR1', 'LASSI_B_CR2',
                    'COMBINED_NE4S', 'AMYLPET',
                    'PTAU_217_CONCNTRTN',
                    'FL_UDSD']

    df = df[df['FL_UDSD'] != "Unknown"]

    df = df[feature_cols]

    metadata = Metadata.detect_from_dataframe(df)
    # metadata.update_column(column_name='PTID', sdtype='categorical')

    # Set CDRGLOB as categorical
    metadata.update_column(
        column_name='CDRGLOB',
        sdtype='categorical'
    )

    metadata.save_to_json("metadata.json", mode="overwrite")
    CONFIGS = {
        "baseline": {
            "epochs": 500,
            "batch_size": 100,
            "generator_dim": (128, 128),
            "discriminator_dim": (128, 128),
            "embedding_dim": 128,
            "generator_lr": 2e-4,
            "discriminator_lr": 2e-4,
        },
        "small_model": {
            "epochs": 300,
            "batch_size": 50,
            "generator_dim": (64, 64),
            "discriminator_dim": (64, 64),
            "embedding_dim": 64,
            "generator_lr": 2e-4,
            "discriminator_lr": 2e-4,
        },
        "large_model": {
            "epochs": 500,
            "batch_size": 100,
            "generator_dim": (256, 256),
            "discriminator_dim": (256, 256),
            "embedding_dim": 256,
            "generator_lr": 2e-4,
            "discriminator_lr": 2e-4,
        },
        "deep_model": {
            "epochs": 500,
            "batch_size": 100,
            "generator_dim": (128, 128, 128),
            "discriminator_dim": (128, 128, 128),
            "embedding_dim": 128,
            "generator_lr": 2e-4,
            "discriminator_lr": 2e-4,
        },
        "high_lr": {
            "epochs": 500,
            "batch_size": 100,
            "generator_dim": (128, 128),
            "discriminator_dim": (128, 128),
            "embedding_dim": 128,
            "generator_lr": 5e-4,
            "discriminator_lr": 5e-4,
        },
        "large_batch": {
            "epochs": 500,
            "batch_size": 200,
            "generator_dim": (128, 128),
            "discriminator_dim": (128, 128),
            "embedding_dim": 128,
            "generator_lr": 2e-4,
            "discriminator_lr": 2e-4,
        },
    }

    results = []

    for config_name, config in CONFIGS.items():
        print(f"Training CTGAN with configuration: {config_name}")
        synthesizer = train_synthesizer(df, metadata, config, enable_gpu=True)
        synthetic_data = synthesizer.sample(len(df))

        # Post-process CDRSUM to ensure 0.5 increments
        synthetic_data['CDRSUM'] = synthetic_data['CDRSUM'].apply(round_to_half)

        quality_report = evaluate_quality(df, synthetic_data, metadata)

        joblib.dump(synthesizer, f"models/ctgan_model_{config_name}.joblib")
        
        # Collect all metrics
        metrics = {
            'config_name': config_name,
            'overall_score': quality_report.get_score(),
            'column_shapes_score': quality_report.get_details(property_name='Column Shapes')['Score'].mean(),
            'column_pair_trends_score': quality_report.get_details(property_name='Column Pair Trends')['Score'].mean()
        }
        results.append(metrics)

    # Save all results at once
    results_df = pd.DataFrame(results)
    results_df.to_csv("quality_reports.csv", index=False)
