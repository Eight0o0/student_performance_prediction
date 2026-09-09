import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    A comprehensive model training class that handles:
    - Multiple model initialization
    - Cross-validation training
    - Hyperparameter tuning
    - Model persistence
    """
    
    def __init__(self, problem_type='classification', random_state=42):
        """
        Initialize the ModelTrainer
        
        Parameters:
        -----------
        problem_type : str
            'classification' or 'regression'
        random_state : int
            Random seed for reproducibility
        """
        self.problem_type = problem_type
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf
        self.results = {}
        self.training_history = {}
        
    def create_models(self):
        """
        Initialize models based on problem type
        
        Returns:
        --------
        dict: Dictionary of initialized models
        """
        if self.problem_type == 'classification':
            self.models = {
                'Logistic Regression': LogisticRegression(
                    max_iter=1000, 
                    random_state=self.random_state,
                    class_weight='balanced'  # Handles class imbalance
                ),
                'Random Forest': RandomForestClassifier(
                    n_estimators=100, 
                    random_state=self.random_state,
                    class_weight='balanced'
                ),
                'XGBoost': XGBClassifier(
                    random_state=self.random_state,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    scale_pos_weight=1  # Adjust if imbalanced
                )
            }
            print(f"✅ Created {len(self.models)} classification models")
            
        else:  # regression
            self.models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(
                    n_estimators=100, 
                    random_state=self.random_state
                ),
                'XGBoost': XGBRegressor(
                    random_state=self.random_state,
                    objective='reg:squarederror'
                )
            }
            print(f"✅ Created {len(self.models)} regression models")
        
        return self.models
    
    def train_models(self, X_train, y_train, cv_folds=5, verbose=True):
        """
        Train all models with cross-validation
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        cv_folds : int
            Number of cross-validation folds
        verbose : bool
            Print progress
            
        Returns:
        --------
        object: Best performing model
        """
        print("\n" + "="*60)
        print("TRAINING MODELS WITH CROSS-VALIDATION")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")
            
            try:
                # Choose scoring metric based on problem type
                if self.problem_type == 'classification':
                    scoring = 'accuracy'
                    scores = cross_val_score(
                        model, X_train, y_train, 
                        cv=cv_folds, scoring=scoring
                    )
                    mean_score = scores.mean()
                    std_score = scores.std()
                    
                    if verbose:
                        print(f"   CV Accuracy: {mean_score:.4f} (+/- {std_score:.4f})")
                        print(f"   Individual folds: {scores}")
                
                else:  # regression
                    scoring = 'r2'
                    scores = cross_val_score(
                        model, X_train, y_train, 
                        cv=cv_folds, scoring=scoring
                    )
                    mean_score = scores.mean()
                    std_score = scores.std()
                    
                    if verbose:
                        print(f"   CV R² Score: {mean_score:.4f} (+/- {std_score:.4f})")
                        print(f"   Individual folds: {scores}")
                
                # Fit on full training data
                model.fit(X_train, y_train)
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'cv_mean': mean_score,
                    'cv_std': std_score,
                    'scores': scores,
                    'is_fitted': True
                }
                
                # Track best model
                if mean_score > self.best_score:
                    self.best_score = mean_score
                    self.best_model = model
                    self.best_model_name = name
                    
            except Exception as e:
                print(f"   ❌ Error training {name}: {e}")
                self.results[name] = {
                    'model': model,
                    'cv_mean': -np.inf,
                    'cv_std': 0,
                    'scores': [],
                    'is_fitted': False,
                    'error': str(e)
                }
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"🏆 Best model: {self.best_model_name}")
        print(f"🏆 Best CV score: {self.best_score:.4f}")
        
        return self.best_model
    
    def hyperparameter_tuning(self, X_train, y_train, model_type='xgboost', 
                              cv_folds=3, n_iter=10, verbose=True):
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        model_type : str
            'xgboost' or 'random_forest'
        cv_folds : int
            Number of cross-validation folds for tuning
        n_iter : int
            Number of iterations for RandomizedSearch (not used in GridSearch)
        verbose : bool
            Print progress
            
        Returns:
        --------
        object: Tuned best model
        """
        print("\n" + "="*60)
        print(f"HYPERPARAMETER TUNING - {model_type.upper()}")
        print("="*60)
        
        if model_type.lower() == 'xgboost':
            if self.problem_type == 'classification':
                model = XGBClassifier(
                    random_state=self.random_state,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.7, 0.8, 1.0],
                    'colsample_bytree': [0.7, 0.8, 1.0],
                    'min_child_weight': [1, 3, 5]
                }
                scoring = 'accuracy'
            else:
                model = XGBRegressor(
                    random_state=self.random_state,
                    objective='reg:squarederror'
                )
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.7, 0.8, 1.0],
                    'colsample_bytree': [0.7, 0.8, 1.0],
                    'min_child_weight': [1, 3, 5]
                }
                scoring = 'r2'
            
            print(f"🔍 Searching over {len(param_grid)} parameter combinations...")
            
            grid_search = GridSearchCV(
                model, 
                param_grid, 
                cv=cv_folds, 
                scoring=scoring,
                n_jobs=-1,  # Use all CPU cores
                verbose=1 if verbose else 0
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"\n✅ Best parameters: {grid_search.best_params_}")
            print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
            
            self.best_model = grid_search.best_estimator_
            self.best_model_name = f"XGBoost (Tuned)"
            self.best_score = grid_search.best_score_
            
            # Store tuning results
            self.training_history['tuning'] = {
                'model_type': 'xgboost',
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
        elif model_type.lower() == 'random_forest':
            if self.problem_type == 'classification':
                model = RandomForestClassifier(
                    random_state=self.random_state,
                    class_weight='balanced'
                )
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
                scoring = 'accuracy'
            else:
                model = RandomForestRegressor(
                    random_state=self.random_state
                )
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
                scoring = 'r2'
            
            print(f"🔍 Searching over {len(param_grid)} parameter combinations...")
            
            grid_search = GridSearchCV(
                model, 
                param_grid, 
                cv=cv_folds, 
                scoring=scoring,
                n_jobs=-1,
                verbose=1 if verbose else 0
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"\n✅ Best parameters: {grid_search.best_params_}")
            print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
            
            self.best_model = grid_search.best_estimator_
            self.best_model_name = f"Random Forest (Tuned)"
            self.best_score = grid_search.best_score_
            
            # Store tuning results
            self.training_history['tuning'] = {
                'model_type': 'random_forest',
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
        
        else:
            print(f"❌ Unknown model type: {model_type}")
            print("   Available: 'xgboost', 'random_forest'")
        
        return self.best_model
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance from the best model
        
        Parameters:
        -----------
        feature_names : list
            List of feature names
            
        Returns:
        --------
        pd.DataFrame: Feature importance dataframe
        """
        if self.best_model is None:
            print("❌ No model trained yet")
            return None
        
        # Check if model supports feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(len(importances))]
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\n📊 Feature Importance (Top 10):")
            print(importance_df.head(10).to_string(index=False))
            
            return importance_df
        else:
            print("⚠️ Model doesn't support feature importance")
            return None
    
    def predict(self, X):
        """
        Make predictions using the best model
        
        Parameters:
        -----------
        X : array-like
            Features to predict on
            
        Returns:
        --------
        array: Predictions
        """
        if self.best_model is None:
            print("❌ No model trained yet")
            return None
        
        return self.best_model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities (classification only)
        
        Parameters:
        -----------
        X : array-like
            Features to predict on
            
        Returns:
        --------
        array: Prediction probabilities
        """
        if self.problem_type != 'classification':
            print("⚠️ predict_proba only available for classification")
            return None
        
        if self.best_model is None:
            print("❌ No model trained yet")
            return None
        
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X)
        else:
            print("⚠️ Model doesn't support predict_proba")
            return None
    
    def save_model(self, filepath='models/best_model.pkl'):
        """
        Save the best model to disk
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
            
        Returns:
        --------
        str: Path where model was saved
        """
        if self.best_model is None:
            print("❌ No model to save")
            return None
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        joblib.dump(self.best_model, filepath)
        print(f"✅ Model saved to {filepath}")
        
        # Also save training results
        results_path = filepath.replace('.pkl', '_results.pkl')
        results_data = {
            'model_name': self.best_model_name,
            'best_score': self.best_score,
            'problem_type': self.problem_type,
            'results': self.results,
            'training_history': self.training_history
        }
        joblib.dump(results_data, results_path)
        print(f"✅ Training results saved to {results_path}")
        
        return filepath
    
    def load_model(self, filepath='models/best_model.pkl'):
        """
        Load a saved model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        object: Loaded model
        """
        try:
            self.best_model = joblib.load(filepath)
            print(f"✅ Model loaded from {filepath}")
            return self.best_model
        except FileNotFoundError:
            print(f"❌ Model file not found: {filepath}")
            return None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def get_model_summary(self):
        """
        Get a summary of the training results
        
        Returns:
        --------
        dict: Summary dictionary
        """
        summary = {
            'problem_type': self.problem_type,
            'best_model_name': self.best_model_name,
            'best_cv_score': self.best_score,
            'total_models_trained': len(self.results),
            'models': {}
        }
        
        for name, result in self.results.items():
            summary['models'][name] = {
                'cv_mean': result.get('cv_mean', None),
                'cv_std': result.get('cv_std', None),
                'is_fitted': result.get('is_fitted', False)
            }
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary of training results"""
        print("\n" + "="*60)
        print("MODEL TRAINING SUMMARY")
        print("="*60)
        
        print(f"\n📊 Problem Type: {self.problem_type}")
        print(f"🏆 Best Model: {self.best_model_name}")
        print(f"🏆 Best CV Score: {self.best_score:.4f}")
        
        print("\n📈 All Models Performance:")
        print("-"*40)
        for name, result in self.results.items():
            if result.get('is_fitted', False):
                print(f"  {name}: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")
            else:
                print(f"  {name}: Failed to train")
        
        if 'tuning' in self.training_history:
            print(f"\n🔧 Tuning Results:")
            print(f"  Best Parameters: {self.training_history['tuning']['best_params']}")
        
        print("\n" + "="*60)


# Example usage
if __name__ == "__main__":
    # This is just for testing the class
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test the trainer
    trainer = ModelTrainer(problem_type='classification')
    trainer.create_models()
    trainer.train_models(X_train, y_train, cv_folds=3)
    trainer.hyperparameter_tuning(X_train, y_train, model_type='xgboost', cv_folds=2)
    trainer.print_summary()