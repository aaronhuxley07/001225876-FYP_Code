**Project Title:** Cryptocurrency Fraud Detection using Machine Learning (ML) and Explainable AI (XAI)  
**Name:** Aaron Huxley Temcykumar
**Student ID:** 001225876  
**Repository:** 001225876-FYP_Code  

---

## Overview

This project aims to detect fraudulent cryptocurrency transactions using machine learning models and interpret their predictions using Explainable AI (SHAP). The solution includes a training pipeline, model optimisation, SHAP-based explanation, and a Streamlit web interface for testing and user interaction.

---

## Project Structure

```
001225876-FYP_Code/
├── application/
│   ├── main.py            # Entry point
│   ├── controller.py      # Program Flow logic
│   ├── model.py           # Load model + prediction logic
│   ├── view.py            # Streamlit display logic
├── notebooks/
│   ├── machine_learning_pipeline.ipynb     # Preprocessing, training, fine-tuning, evaluation
│   └── sample_dataset_creator.ipynb        # Creates sample datasets for web testing
├── pdfs_of_notebooks/
│   ├── machine_learning_pipeline.pdf
│   └── sample_dataset_creator.pdf
├── datasets/
│   ├── elliptic_txs_classes.csv            # Need to be downloaded from https://www.kaggle.com/datasets/ellipticco/elliptic-data-set due to GitHub file size limits
│   ├── elliptic_txs_features.csv           # Need to be downloaded from https://www.kaggle.com/datasets/ellipticco/elliptic-data-set due to GitHub file size limits
│   └── sample_deployment_data.csv
├── models/
│   ├── model.pkl                           # Trained ML model
│   └── explainer.pkl                       # SHAP explainer object
├── README.md
└── requirements.txt
```

---

## Notebooks Summary

### `machine_learning_pipeline.ipynb`
- Loads Elliptic dataset
- Preprocesses data (scaling, splitting)
- Trains and fine-tunes ML models (Logistic Regression, Random Forest, XGBoost)
- Evaluates and compares models
- Saves the best model (`model.pkl`) and SHAP explainer (`explainer.pkl`)

### `sample_dataset_creator.ipynb`
- Generates smaller datasets with real transaction data for quick testing in the Streamlit app

> PDF versions of both notebooks are provided in `/pdfs_of_notebooks/` for easy viewing without running lengthy training processes.

---

## Streamlit Web App

### `crypto_fraud_detection.py`
- Loads the trained `model.pkl` and `explainer.pkl`
- Lets users upload sample CSV datasets
- Predicts if a transaction is licit or illicit
- Allows for export of predictions
- Visualises explanations using SHAP
- Allows for export of Explanations
- Visualises SHAP Heatmap (Can be toggled on/off)
- Allows for export of SHAP Heatmap

---

## Run Instructions

### 📊 Dataset Info

The original Elliptic dataset used in this project is publicly available on Kaggle:

[Elliptic Dataset on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

Please download the following files from the Kaggle page and place them in the `datasets/` folder before running the notebooks or app:

- `elliptic_txs_classes.csv`
- `elliptic_txs_features.csv`

---

For convenience, sample smaller datasets used for testing the deployed Streamlit app are included in this repository under the `datasets/` folder.

### Run notebooks in Google Colab

Click badges to open notebooks in Google Colab:

- [![Open ML Pipeline](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aaronhuxley07/001225876-FYP_Code/blob/main/notebooks/machine_learning_pipeline.ipynb)
- [![Open Sample Creator](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aaronhuxley07/001225876-FYP_Code/blob/main/notebooks/sample_dataset_creator.ipynb)

### Run application locally

Clone repo
```bash
git clone https://github.com/aaronhuxley07/001225876-FYP_Code.git
cd 001225876-FYP_Code
```

Install dependencies
```bash
pip install -r requirements.txt
```

Launch Streamlit app
```bash
source venv/bin/activate
python -m run streamlit application/main.py
```
