import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .pass {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .fail {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>🎓 Student Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("Predict whether a student is likely to pass or fail based on their academic and personal factors.")

# Load model
@st.cache_resource
def load_model():
    try:
        model_path = 'models/best_model.pkl'
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            st.error("Model not found. Please train the model first by running main.py")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.warning("⚠️ Model not loaded. Please run the training pipeline first.")
    st.stop()

# Sidebar for feature input
st.sidebar.title("📝 Student Information")
st.sidebar.markdown("Enter the student's details below:")

# Create input fields
def create_inputs():
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=15, max_value=22, value=17)
        sex = st.selectbox("Gender", ["Female", "Male"])
        address = st.selectbox("Address Type", ["Urban", "Rural"])
        famsize = st.selectbox("Family Size", ["≤ 3", "> 3"])
        Pstatus = st.selectbox("Parent Cohabitation", ["Living Together", "Apart"])
        Medu = st.slider("Mother's Education", 0, 4, 2)
        Fedu = st.slider("Father's Education", 0, 4, 2)
        Mjob = st.selectbox("Mother's Job", ["teacher", "health", "services", "at_home", "other"])
        Fjob = st.selectbox("Father's Job", ["teacher", "health", "services", "at_home", "other"])
    
    with col2:
        traveltime = st.slider("Travel Time to School (minutes)", 1, 4, 2)
        studytime = st.slider("Weekly Study Time (hours)", 1, 4, 2)
        schoolsup = st.selectbox("School Support", ["No", "Yes"])
        famsup = st.selectbox("Family Support", ["No", "Yes"])
        paid = st.selectbox("Paid Tutoring", ["No", "Yes"])
        activities = st.selectbox("Extracurricular Activities", ["No", "Yes"])
        nursery = st.selectbox("Attended Nursery", ["No", "Yes"])
        higher = st.selectbox("Wants Higher Education", ["No", "Yes"])
        internet = st.selectbox("Internet Access at Home", ["No", "Yes"])
    
    # More inputs in a single column
    romantic = st.sidebar.selectbox("In a Romantic Relationship", ["No", "Yes"])
    famrel = st.sidebar.slider("Family Relationship Quality", 1, 5, 4)
    freetime = st.sidebar.slider("Free Time (1=low, 5=high)", 1, 5, 3)
    goout = st.sidebar.slider("Going Out with Friends (1=low, 5=high)", 1, 5, 3)
    Dalc = st.sidebar.slider("Workday Alcohol Consumption