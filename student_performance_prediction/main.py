#!/usr/bin/env python
"""
Student Performance Prediction - Main Pipeline
Run this file to execute the entire ML pipeline
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from train_model import ModelTrainer
from evaluate_model import ModelEvaluator
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

def main():
    print("="*60)
    print("🎓 STUDENT PERFORMANCE PREDICTION PIPELINE")
    print("="*60)
    
    # Configuration
    DATA_PATH = 'data/student-mat.csv'
    PROBLEM_TYPE = 'classification'  # 'classification' or 'regression'
    
    # Step 1: Data Preprocessing
    print("\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    
    preprocessor = DataPreprocessor(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocessor.run_full_pipeline(target_type=PROBLEM_TYPE)
    
    # Step 2: Feature Engineering
    print("\n" + "="*60)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*60)
    
    # Convert back to DataFrame for feature engineering
    # Note: Since we used ColumnTransformer, we need to get original data for feature engineering
    # For simplicity, we'll apply feature engineering before preprocessing
    # Let's reload and process differently for feature engineering
    
    print("⚠️ Re-running pipeline with feature engineering...")
    # Reload raw data
    df = pd.read_csv(DATA_PATH, sep=';')
    
    # Apply feature engineering
    feature_engineer = FeatureEngineer(add_effort_score=True, add_risk_flags=True)
    df_engineered = feature_engineer.transform(df)
    
    # Now preprocess again with engineered features
    preprocessor2 = DataPreprocessor(DATA_PATH)
    preprocessor2.df = df_engineered
    X_train, X_test, y_train, y_test = preprocessor2.run_full_pipeline(target_type=PROBLEM_TYPE)
    
    # Step 3: Model Training
    print("\n" + "="*60)
    print("STEP 3: MODEL TRAINING")
    print("="*60)
    
    trainer = ModelTrainer(problem_type=PROBLEM_TYPE)
    trainer.create_models()
    trainer.train_models(X_train, y_train)
    
    # Optional: Hyperparameter tuning
    print("\n" + "="*60)
    print("STEP 4: HYPERPARAMETER TUNING")
    print("="*60)
    trainer.hyperparameter_tuning(X_train, y_train, model_type='xgboost')
    
    # Save the model
    trainer.save_model('models/best_model.pkl')
    
    # Step 5: Model Evaluation
    print("\n" + "="*60)
    print("STEP 5: MODEL EVALUATION")
    print("="*60)
    
    evaluator = ModelEvaluator(trainer.best_model, X_test, y_test, problem_type=PROBLEM_TYPE)
    metrics, cm = evaluator.evaluate()
    
    # Visualizations
    evaluator.plot_confusion_matrix('models/confusion_matrix.png')
    evaluator.plot_roc_curve('models/roc_curve.png')
    
    # Feature importance
    # Get feature names from preprocessor
    cat_features = preprocessor.categorical_cols
    num_features = preprocessor.numerical_cols
    
    # For OHE features, we need to get the actual feature names
    feature_names = []
    if hasattr(preprocessor.preprocessor, 'transformers_'):
        for name, transformer, columns in preprocessor.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat':
                # Get OHE feature names
                if hasattr(transformer, 'get_feature_names_out'):
                    ohe_names = transformer.get_feature_names_out(columns)
                    feature_names.extend(ohe_names)
                else:
                    feature_names.extend(columns)
    
    if len(feature_names) == X_train.shape[1]:
        evaluator.plot_feature_importance(feature_names, top_n=15, save_path='models/feature_importance.png')
    else:
        evaluator.plot_feature_importance(top_n=15, save_path='models/feature_importance.png')
    
    # Step 6: Summary
    print("\n" + "="*60)
    print("📊 PIPELINE SUMMARY")
    print("="*60)
    print(f"✅ Problem Type: {PROBLEM_TYPE}")
    print(f"✅ Best Model: {trainer.best_model_name}")
    print(f"✅ Best CV Score: {trainer.best_score:.4f}")
    
    if PROBLEM_TYPE == 'classification':
        print(f"✅ Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"✅ Test F1 Score: {metrics['f1']:.4f}")
    else:
        print(f"✅ Test R² Score: {metrics['r2']:.4f}")
        print(f"✅ Test RMSE: {metrics['rmse']:.4f}")
    
    print("\n✅ Pipeline completed successfully!")
    print("📁 Check the 'models/' directory for saved artifacts")

if __name__ == "__main__":
    main()