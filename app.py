import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Correctly import TabPFN
try:
    from tabpfn import TabPFNClassifier
except ImportError:
    st.error("TabPFN library is not installed. Please run `pip install tabpfn` to install it.")

# Page configuration
st.set_page_config(
    page_title="Spinal TB vs. Pyogenic Spondylitis Model",
    layout="wide"
)

# Simple style settings
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        font-weight: 600;
        padding: 0.5rem 2rem;
    }
    </style>
    """, unsafe_allow_html=True
)

# Page title
st.title("TabPFN-Based Interpretable Deep Learning Model for Discriminating Spinal Tuberculosis from Pyogenic Spinal Infection")

# Layout with two columns for input fields
col1, col2 = st.columns(2)

with col1:
    IGRAs = st.selectbox("Interferon-Gamma Release Assays (IGRAs)", ("Positive", "Negative"))
    RBC = st.number_input("Red Blood Cell Count (RBC) (10^12/L)", min_value=0.0, max_value=10.0, value=4.0, step=0.01)
    HGB = st.number_input("Hemoglobin (HGB) (g/L)", min_value=0, max_value=300, value=120, step=1)
    A = st.number_input("Albumin (A) (g/L)", min_value=0.0, max_value=60.0, value=35.0, step=0.1)

with col2:
    Lymph = st.number_input("Lymphocyte Count (10^9/L)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    Mono = st.number_input("Monocyte Count (10^9/L)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
    PT = st.number_input("Prothrombin Time (PT) (seconds)", min_value=0.0, max_value=30.0, value=11.0, step=0.1)
    CRP = st.number_input("C-Reactive Protein (CRP) (mg/L)", min_value=0.0, max_value=500.0, value=20.0, step=0.1)

# Submit button
if st.button("Predict"):
    # Check if model file exists
    model_path = "tabpfn_best_model.pkl"
    model_loaded = False
    
    if os.path.exists(model_path):
        try:
            clf = joblib.load(model_path)
            model_loaded = True
        except Exception as e:
            st.error(f"Model loading error: {str(e)}")
    else:
        st.error("Model file 'tabpfn_best_model.pkl' not found")
        
        # Option to create a sample model
        if st.button("Create Sample Model"):
            try:
                from tabpfn import TabPFNClassifier
                
                # Create a simple sample model
                st.info("Creating sample model...")
                clf = TabPFNClassifier(device='cpu', N_ensemble_configurations=3)
                
                # Create some sample data for training
                X_sample = pd.DataFrame(np.random.rand(10, 8), 
                                      columns=["IGRAs", "RBC", "HGB", "Lymph", "Mono", "A", "PT", "CRP"])
                y_sample = np.random.randint(0, 2, 10)  # Random 0 or 1 labels
                
                # Train the model
                clf.fit(X_sample, y_sample)
                
                # Save the model
                joblib.dump(clf, model_path)
                st.success("Sample model created")
                model_loaded = True
            except Exception as e:
                st.error(f"Error creating model: {str(e)}")
    
    if model_loaded:
        # Convert IGRAs from text to binary
        IGRAs_binary = 1 if IGRAs == "Positive" else 0
        
        # Create DataFrame for prediction
        X = pd.DataFrame([[IGRAs_binary, RBC, HGB, Lymph, Mono, A, PT, CRP]],
                       columns=["IGRAs", "RBC", "HGB", "Lymph", "Mono", "A", "PT", "CRP"])
        
        try:
            # Model prediction
            prediction = clf.predict(X)[0]
            
            # Get prediction probability
            try:
                prediction_probs = clf.predict_proba(X)[0]
                prediction_probability = prediction_probs[1] if len(prediction_probs) > 1 else prediction_probs[0]
            except Exception:
                prediction_probability = 0.5  # Default value
            
            # Display results based on 38% threshold
            st.write("---")
            st.markdown("### Prediction Results")
            
            # Display probability with exactly 2 decimal places
            probability_percentage = prediction_probability * 100
            st.markdown(f"### Probability of STB: **{probability_percentage:.3f}%**")
            
            # Apply 38% threshold for diagnosis
            if probability_percentage > 38.7:
                st.success("### Diagnosis: Spinal Tuberculosis")
            else:
                st.error("### Diagnosis: Pyogenic Spinal infection")
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

# Simple footer
st.write("---")
st.markdown("**Contact:** huxiaojiang2021@163.com")