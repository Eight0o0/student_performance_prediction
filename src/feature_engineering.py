import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom feature engineering transformer for student data"""
    
    def __init__(self, add_effort_score=True, add_risk_flags=True):
        self.add_effort_score = add_effort_score
        self.add_risk_flags = add_risk_flags
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Apply feature engineering to the DataFrame"""
        X = X.copy()
        
        # 1. Effort Score: Combine study time and attendance
        if self.add_effort_score and 'studytime' in X.columns and 'absences' in X.columns:
            # Invert absences (more absences = lower effort)
            max_absences = X['absences'].max()
            attendance_score = 1 - (X['absences'] / max_absences)
            # Normalize study time (scale 1-4)
            study_score = X['studytime'] / 4
            X['effort_score'] = (study_score * 0.6 + attendance_score * 0.4)
            print("✅ Added 'effort_score' feature")
        
        # 2. Risk Flags
        if self.add_risk_flags:
            # High absence risk (> 20 absences)
            if 'absences' in X.columns:
                X['high_absence_risk'] = (X['absences'] > 20).astype(int)
                print("✅ Added 'high_absence_risk' feature")
            
            # Past failure history
            if 'failures' in X.columns:
                X['has_past_failures'] = (X['failures'] > 0).astype(int)
                print("✅ Added 'has_past_failures' feature")
                
            # Support score: Combine family and school support
            support_cols = []
            if 'famsup' in X.columns:
                X['famsup_encoded'] = (X['famsup'] == 'yes').astype(int)
                support_cols.append('famsup_encoded')
            if 'schoolsup' in X.columns:
                X['schoolsup_encoded'] = (X['schoolsup'] == 'yes').astype(int)
                support_cols.append('schoolsup_encoded')
            
            if support_cols:
                X['support_score'] = X[support_cols].mean(axis=1)
                print("✅ Added 'support_score' feature")
        
        # 3. Age groups
        if 'age' in X.columns:
            bins = [0, 16, 18, 20, 100]
            labels = ['young', 'medium', 'older', 'mature']
            X['age_group'] = pd.cut(X['age'], bins=bins, labels=labels)
            print("✅ Added 'age_group' feature")
        
        # 4. Weekend vs Weekday alcohol consumption difference
        if 'Dalc' in X.columns and 'Walc' in X.columns:
            X['alc_diff'] = X['Walc'] - X['Dalc']
            print("✅ Added 'alc_diff' feature")
        
        # 5. Absence rate (absences / classes)
        if 'absences' in X.columns:
            # Assuming 32 class days (standard in this dataset)
            X['absence_rate'] = X['absences'] / 32
            print("✅ Added 'absence_rate' feature")
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names for sklearn compatibility"""
        return self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None