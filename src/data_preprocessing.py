import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.categorical_cols = None
        self.numerical_cols = None
        self.preprocessor = None
        
    def load_data(self):
        """Load the UCI Student Performance dataset"""
        # Using the Math course dataset
        self.df = pd.read_csv(self.data_path, sep=';')
        print(f"✅ Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(f"Columns: {self.df.columns.tolist()}")
        return self.df
    
    def clean_data(self):
        """Handle missing values and basic cleaning"""
        # Check for missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(f"⚠️ Missing values found: {missing[missing > 0]}")
            # Fill numerical missing with median, categorical with mode
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                else:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
        else:
            print("✅ No missing values found")
        
        # Remove duplicates if any
        initial_len = len(self.df)
        self.df.drop_duplicates(inplace=True)
        if len(self.df) < initial_len:
            print(f"✅ Removed {initial_len - len(self.df)} duplicate rows")
        
        # Fix inconsistent categorical values
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.df[col] = self.df[col].str.lower().str.strip()
        
        return self.df
    
    def create_target(self, target_type='classification'):
        """
        Create target variable
        target_type: 'regression' -> G3 (0-20)
                     'classification' -> Pass/Fail (1 if G3 >= 10 else 0)
        """
        if target_type == 'regression':
            self.y = self.df['G3'].values
            print(f"✅ Regression target: G3 (range: {self.y.min()} to {self.y.max()})")
        else:  # classification
            self.y = (self.df['G3'] >= 10).astype(int).values
            pass_count = np.sum(self.y)
            fail_count = len(self.y) - pass_count
            print(f"✅ Classification target: Pass={pass_count}, Fail={fail_count}")
        
        # Drop target from features
        self.X = self.df.drop(['G3'], axis=1)
        return self.X, self.y
    
    def identify_feature_types(self):
        """Separate numerical and categorical features"""
        self.categorical_cols = self.X.select_dtypes(include=['object']).columns.tolist()
        self.numerical_cols = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"✅ Categorical features: {len(self.categorical_cols)}")
        print(f"✅ Numerical features: {len(self.numerical_cols)}")
        return self.categorical_cols, self.numerical_cols
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Create preprocessing pipeline and split data
        """
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y if len(np.unique(self.y)) == 2 else None
        )
        print(f"✅ Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")
        
        # Create preprocessing pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        
        # Numerical pipeline
        numerical_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine pipelines
        self.preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, self.numerical_cols),
            ('cat', categorical_pipeline, self.categorical_cols)
        ])
        
        # Fit and transform training data
        self.X_train_processed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_processed = self.preprocessor.transform(self.X_test)
        
        print(f"✅ Processed train shape: {self.X_train_processed.shape}")
        print(f"✅ Processed test shape: {self.X_test_processed.shape}")
        
        return self.X_train_processed, self.X_test_processed, self.y_train, self.y_test
    
    def run_full_pipeline(self, target_type='classification'):
        """Run all preprocessing steps"""
        self.load_data()
        self.clean_data()
        self.create_target(target_type)
        self.identify_feature_types()
        return self.preprocess_data()