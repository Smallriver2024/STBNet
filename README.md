# eSpineTBNET
# TabPFN-Based Interpretable Deep Learning Model for Discriminating Spinal Tuberculosis from Pyogenic Spinal Infection

This repository contains the code and materials supporting the SCI manuscript titled **"TabPFN-Based Interpretable Deep Learning Model for Discriminating Spinal Tuberculosis from Pyogenic Spinal Infection"**.

## Overview

In this project, we implemented the TabPFN (Tabular Prior-Data Fitting Network) model to effectively differentiate between spinal tuberculosis (STB) and pyogenic spinal infection (PSI) based on serological biomarkers. Model interpretation was enhanced using SHAP (SHapley Additive exPlanations) analysis, providing insights into feature importance and interactions.

Key contributions include:
- Training and evaluation of TabPFN models.
- SHAP-based feature importance and dependence plots.
- Performance metrics visualization through ROC curves and confusion matrices.
- An interactive Streamlit web application for user-friendly predictions and visualizations.

![image](https://github.com/user-attachments/assets/90250a0b-cddf-4aa6-ab4f-2e334f8b42da)

  
Figure 1. Workflow of model development
(A) Population and Variables: A total of 342 patients were divided into a training set (n = 217) and validation set (n = 125). Clinical variables including basic information, laboratory examination results, and IGRA levels were collected to form the feature matrix. (B) mNGS-Based Labels: Specimens were obtained through biopsy or surgery and subjected to mNGS for pathogen identification. Patients were labeled as STB or PSI based on sequencing results. (C) Model Development: LASSO regression was applied for feature selection, followed by hyperparameter tuning using five-fold cross-validation. The final model was trained using the full training set. (D) Evaluation and Explanation: Model performance was assessed by ROC curves and confusion matrices. SHAP summary plots were used to interpret feature contributions. The trained model was deployed via a user-friendly TabPFN-based web application.

## Environment Setup

The project was developed and tested using **Python 3.10**. To set up the environment, follow these steps:

1. Create and activate a virtual environment:
```bash
conda create -n tabpfn_env python=3.10
conda activate tabpfn_env
```

2. Install the necessary dependencies:
```bash
pip install numpy pandas matplotlib seaborn shap scikit-learn tabpfn notebook streamlit xgboost lightgbm
```

## Dependencies

The essential Python libraries used in this project include:
- **numpy**
- **pandas**
- **matplotlib**
- **seaborn** (optional, for additional visualizations)
- **shap** (for feature interpretability)
- **scikit-learn**
- **tabpfn** (core predictive model)
- **notebook** (Jupyter notebook)
- **streamlit** (interactive web application)

## Streamlit Web Application

The Streamlit app allows interactive prediction and visualization of results:

To launch the Streamlit app, run:
```bash
streamlit run app.py
```

Open the provided local URL in your web browser to access the application interface.

## Usage

Detailed steps for reproducing the analysis and visualizations are provided in the included Jupyter notebook (`TabPFN.ipynb`) and Python scripts (`app.py`, `shap_dependence_plot.py`, `shapplot1.py`, `shapvalue.py`, `Confusion_matrix_ROC.py`).

Feel free to reach out if you encounter issues or have questions about the analysis pipeline.

