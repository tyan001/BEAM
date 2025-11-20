# 🧠 BEAM-ML - Blood & Exam Assessment Model using Machine Learning
> Predicting cognitive decline through AI-powered analysis of blood biomarkers and cognitive assessments

## 📋 Overview

BEAM-ML (Blood & Exam Assessment Model using Machine Learning) is a machine learning project designed to predict cognitive decline in patients by analyzing:

- 🩸 Blood biomarker data
- 📊 Cognitive examination results
- 🔬 Clinical measurements

Our model aims to enable early detection and intervention for cognitive health issues.

## 🔐 Privacy-First Approach

This repository uses a **synthetic dataset** generated from the original 1FL ADRC dataset using a Generative Adversarial Network (GAN). 

✅ **What this means:**
- Same statistical properties as real data
- Identical data structure
- **Zero risk** to patient privacy
- Fully shareable for research and education

We specifically employed **CTGAN** (Conditional Tabular GAN) for synthetic data generation, ensuring high-quality, realistic samples while maintaining complete confidentiality.

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
# Generate synthetic data
python scripts/CTGAN/generate_data.py
```


### Project Output

Results from feature search and model training are saved in the `results/` directory with detailed performance metrics and model comparisons.

## 🙏 Acknowledgements

- 1FL ADRC for the original dataset structure
- CTGAN developers for synthetic data generation tools

