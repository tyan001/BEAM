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
├── CTGAN/              # 🏭 Synthetic data generation scripts
├── data/               # 📊 Synthetic data
├── scripts/            # 🤖 ML model implementations
```

## 🔬 Methodology

1. **Data Generation**: CTGAN-based synthetic data creation
2. **Feature Engineering**: Biomarker and cognitive score processing
3. **Model Training**: Machine learning model development
4. **Validation**: Performance evaluation and testing

## 🙏 Acknowledgements

- 1FL ADRC for the original dataset structure
- CTGAN developers for synthetic data generation tools

