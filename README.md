# STBNet  
**TabPFN-Based Interpretable Deep Learning Model for Discriminating Spinal Tuberculosis from Pyogenic Spinal Infection**

This repository contains the code and materials supporting the SCI manuscript titled:  
**"TabPFN-Based Interpretable Deep Learning Model for Discriminating Spinal Tuberculosis from Pyogenic Spinal Infection"**

---

## 🧠 Overview

In this project, we implemented the TabPFN (Tabular Prior-Data Fitting Network) model to effectively differentiate between spinal tuberculosis (STB) and pyogenic spinal infection (PSI) based on serological biomarkers. Model interpretation was enhanced using SHAP (SHapley Additive exPlanations) analysis, providing insights into feature importance and interactions.

Key contributions include:
- Training and evaluation of TabPFN models.
- SHAP-based feature importance and dependence plots.
- Performance metrics visualization through ROC curves and confusion matrices.
- An interactive Streamlit web application for user-friendly predictions and visualizations.

![image](https://github.com/user-attachments/assets/90250a0b-cddf-4aa6-ab4f-2e334f8b42da)

**Figure 1. Workflow of model development**  
(A) Population and Variables: A total of 342 patients were divided into a training set (n = 217) and validation set (n = 125). Clinical variables including basic information, laboratory examination results, and IGRA levels were collected to form the feature matrix.  
(B) mNGS-Based Labels: Specimens were obtained through biopsy or surgery and subjected to mNGS for pathogen identification. Patients were labeled as STB or PSI based on sequencing results.  
(C) Model Development: LASSO regression was applied for feature selection, followed by hyperparameter tuning using five-fold cross-validation. The final model was trained using the full training set.  
(D) Evaluation and Explanation: Model performance was assessed by ROC curves and confusion matrices. SHAP summary plots were used to interpret feature contributions. The trained model was deployed via a user-friendly TabPFN-based web application.

---

## ⚙️ Environment Setup

This project requires **Python ≥ 3.10**. If you're using Python 3.9 or below, please upgrade your environment before proceeding.

### Step-by-step setup:

1. **Create and activate a virtual environment:**
```bash
conda create -n tabpfn_env python=3.10
conda activate tabpfn_env
```

2. **Install dependencies using `requirements.txt`:**
```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** Please ensure your Python version is above 3.9.  
> 📦 **Important:** The TabPFN model file exceeds GitHub’s 25MB upload limit and is therefore not included in this repository. If you need the pre-trained model file, please contact us via **huxiaojiang2021@163.com**.

---

## 📊 Dependencies

Main Python libraries used:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn` (optional)
- `scikit-learn`
- `shap`
- `tabpfn`
- `notebook`
- `streamlit`

All required packages are listed in `requirements.txt`.

---

## 🌐 Streamlit Web Application

The Streamlit app allows for interactive prediction and visualization of model outputs.

### To launch:
```bash
streamlit run app.py
```
Then open the automatically generated local URL in your browser.

---

## 📁 Usage

Refer to the following files for full workflows:
- `TabPFN.ipynb`: End-to-end pipeline notebook
- `app.py`: Streamlit frontend
- `shapplot1.py`, `shap_dependence_plot.py`, `shapvalue.py`: SHAP-based visualization
- `Confusion_matrix_ROC.py`: Evaluation metrics visualization
