# 🧠 BEAM-ML - Blood & Exam Assessment Model using Machine Learning
> Predicting cognitive decline through AI-powered analysis of blood biomarkers and cognitive assessments

## 📋 Overview

BEAM-ML (Blood & Exam Assessment Model using Machine Learning) is a machine learning project designed to predict cognitive decline in patients by analyzing:

- 🩸 Blood biomarker data
- 📊 Cognitive examination results
- 🔬 Clinical measurements

Our goal is to enable early detection and intervention for cognitive health issues through the use of Machine Learning.

## 🔐 Privacy-First Approach

This repository uses a **synthetic dataset** generated from the original 1FL ADRC dataset using a Generative Adversarial Network (GAN). 

✅ **What this means:**
- Same statistical properties as real data
- Identical data structure
- **Zero risk** to patient privacy
- Fully shareable for research and education

We specifically employed **CTGAN** (Conditional Tabular GAN) for synthetic data generation, ensuring high-quality, realistic samples while maintaining complete confidentiality.

## 🎯 Project Goal & Baseline Performance

**Important Context:** This feature search project aims to identify the best feature combinations based on what was achieved with the original dataset:

- 📊 **Original Dataset Size**: ~2,200 samples
- 🎯 **Six-Class Classification**: ~70% F1 score on test data
- ⚡ **Combined five-class classification**: ~80% F1 score on test data

The feature search explores various biomarker and cognitive score combinations to **replicate and optimize these results** using the synthetic dataset. The goal is to find the optimal set of features that can achieve comparable performance while maintaining model interpretability.

## 📁 Repository Structure
```
BEAM-ML/
├── data/               # 📊 Synthetic data
├── scripts/            # 🤖 Jupyter notebooks for prototyping the ML script
    ├── CTGAN/          # 🏭 Synthetic data generation scripts
feature_search.py       # 🔍 Comprehensive feature search script (main)
```

## 🔬 Methodology

1. **Data Generation**: CTGAN-based synthetic data creation
2. **Feature Engineering**: Biomarker and cognitive score processing
3. **Model Training**: Machine learning model development
4. **Validation**: Performance evaluation and testing

## 🚀 How to Use

### Prerequisites
- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd BEAM-ML
   ```

2. **Install dependencies using uv**
   ```bash
   # Install core dependencies
   uv sync

   # Or install with development tools (Jupyter, Ruff)
   uv sync --group dev
   ```

### Running the Feature Search

The main script performs comprehensive feature search to identify the best biomarker combinations for predicting cognitive decline:

```bash
python feature_search.py
```

**What it does:**
- Loads synthetic data from `data/` directory
- Tests various combinations of blood biomarkers and cognitive scores
- Evaluates multiple machine learning models (Random Forest, Logistic Regression)
- Outputs performance metrics (accuracy, balanced accuracy, F1 score)
- Saves results to `results/` directory

### Generating Synthetic Data (Optional)

If you need to generate new synthetic data using CTGAN:

```bash
# Generate synthetic data (preserves original distribution)

python scripts/CTGAN/generate_data.py --model scripts/CTGAN/CTGAN_models/ctgan_model_small_model.pkl --n_samples 10000

# Generate synthetic data (balanced classes)
python generate_data.py --balance -n_samples 10000

```

### Project Output

Results from feature search and model training are saved in the `results/` directory with detailed performance metrics and model comparisons.

## 🌐 Web Application

BEAM-ML includes a Flask web application that provides an interactive interface for predicting cognitive decline based on patient data.

### Training the Model

**Important:** Before running the web application, you must first train a model with desire features (edit in the train_specific_model.py) to generate the required pickle file:

```bash
# Train a Random Forest model (default)
python train_specific_model.py --model rf --output-dir models

# Or train a Logistic Regression model
python train_specific_model.py --model lr --output-dir models
```

**What this does:**
- Trains a classification model using selected features. by default is set to [MMSE, CDRSUM, CDRGLOB, HVLT_DR, LASSI_B_CR2, APOE, AMYLPET]
- Evaluates model performance on train/test splits
- Saves the trained model to `models/rf.pkl` (or `models/lr.pkl` for logistic regression)
- The output folder contains:
  - Trained model information as a pickle file
  - The testing dataset CSV to use as example
  - Feature importance/coefficients

### Running the Web Application

Once you have trained the model and generated the pickle file, you can start the web application:

```bash
python app.py
```

The application will:
- Load the trained model from `models/rf.pkl` using variable MODEL_PATH in app.py
- Start a Flask server on `http://localhost:5000`
- Provide a web interface for entering patient data and getting predictions

**Features:**
- Interactive form for entering biomarker and cognitive test values
- Real-time predictions showing probability distribution across all diagnosis classes
- Support for multiple cognitive decline categories:
  - Normal cognition
  - Subjective Cognitive Decline
  - Impaired Not SCD/MCI
  - Early MCI
  - Late MCI
  - Dementia

**Note:** The web application expects the model pickle file at `models/rf.pkl`. Make sure this file exists before running the app.

## 🙏 Acknowledgements

- 1FL ADRC for the original dataset structure
- CTGAN developers for synthetic data generation tools

