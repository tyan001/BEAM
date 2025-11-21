This folder contains the code for creating synthetic data using CTGAN.

## Files
- **train_ctgan.py**: Training script for the CTGAN model on the original dataset
- **generate_data.py**: Generation script with balanced/unbalanced sampling options
- **model_trainer.ipynb**: Prototype notebook for CTGAN training

## Generating Synthetic Data

### Regular Generation (Preserves Original Distribution)
```bash
python generate_data.py --model ctgan_model_small_model.pkl --n_samples 10000 --output synthetic_data.csv
```

### Balanced Generation (Equal Samples per Class)
```bash
python generate_data.py --model ctgan_model_small_model.pkl --n_samples 10000 --output balanced_data.csv --balanced
```

The `--balanced` flag forces equal distribution across all 6 FL_UDSD classes:
- Normal cognition
- Subjective Cognitive Decline
- Early MCI
- Late MCI
- Impaired Not SCD/MCI
- Dementia

### Additional Options
- `--target_column`: Specify the column to balance (default: FL_UDSD)
- `--model`: Path to the trained CTGAN model file
- `--n_samples`: Total number of samples to generate
- `--output`: Output CSV file name
