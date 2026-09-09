# src/utils.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

def save_dataframe_summary(df, save_path='data/summary.txt'):
    """Save a summary of the dataframe to a text file"""
    with open(save_path, 'w') as f:
        f.write(f"DataFrame Summary - {datetime.now()}\n")
        f.write("="*60 + "\n")
        f.write(f"Shape: {df.shape}\n\n")
        f.write("Column Info:\n")
        f.write("-"*30 + "\n")
        for col in df.columns:
            f.write(f"{col}: {df[col].dtype}, {df[col].nunique()} unique\n")
        f.write("\nMissing Values:\n")
        f.write("-"*30 + "\n")
        missing = df.isnull().sum()
        for col, val in missing.items():
            if val > 0:
                f.write(f"{col}: {val}\n")
        f.write("\nStatistical Summary:\n")
        f.write("-"*30 + "\n")
        f.write(df.describe().to_string())
    print(f"✅ Summary saved to {save_path}")

def load_model_safe(model_path):
    """Safely load a model with error handling"""
    try:
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            print(f"⚠️ Model not found at {model_path}")
            return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def create_feature_names(preprocessor):
    """Extract feature names after preprocessing"""
    feature_names = []
    if hasattr(preprocessor, 'transformers_'):
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat':
                if hasattr(transformer, 'get_feature_names_out'):
                    ohe_names = transformer.get_feature_names_out(columns)
                    feature_names.extend(ohe_names)
                else:
                    feature_names.extend(columns)
    return feature_names