import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
import shap
import joblib

class ModelEvaluator:
    def __init__(self, model, X_test, y_test, problem_type='classification'):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.problem_type = problem_type
        self.predictions = None
        
    def evaluate(self):
        """Generate predictions and compute metrics"""
        self.predictions = self.model.predict(self.X_test)
        
        if self.problem_type == 'classification':
            # For classification, get probabilities if available
            if hasattr(self.model, 'predict_proba'):
                self.probabilities = self.model.predict_proba(self.X_test)[:, 1]
            else:
                self.probabilities = None
            
            metrics = {
                'accuracy': accuracy_score(self.y_test, self.predictions),
                'precision': precision_score(self.y_test, self.predictions),
                'recall': recall_score(self.y_test, self.predictions),
                'f1': f1_score(self.y_test, self.predictions)
            }
            
            if self.probabilities is not None:
                metrics['roc_auc'] = roc_auc_score(self.y_test, self.probabilities)
            
            print("\n📊 Classification Metrics:")
            for metric, value in metrics.items():
                print(f"   {metric}: {value:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, self.predictions)
            print("\n📊 Confusion Matrix:")
            print(f"   [[{cm[0,0]:4d} {cm[0,1]:4d}]")
            print(f"    [{cm[1,0]:4d} {cm[1,1]:4d}]]")
            
            # Detailed classification report
            print("\n📊 Classification Report:")
            print(classification_report(self.y_test, self.predictions, 
                                       target_names=['Fail', 'Pass']))
            
            return metrics, cm
            
        else:  # regression
            metrics = {
                'mae': mean_absolute_error(self.y_test, self.predictions),
                'rmse': np.sqrt(mean_squared_error(self.y_test, self.predictions)),
                'r2': r2_score(self.y_test, self.predictions)
            }
            
            print("\n📊 Regression Metrics:")
            for metric, value in metrics.items():
                print(f"   {metric}: {value:.4f}")
            
            return metrics, None
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix for classification"""
        if self.problem_type != 'classification':
            print("⚠️ Confusion matrix only for classification")
            return
        
        cm = confusion_matrix(self.y_test, self.predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Fail', 'Pass'], 
                    yticklabels=['Fail', 'Pass'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        if save_path:
            plt.savefig(save_path)
            print(f"✅ Confusion matrix saved to {save_path}")
        plt.show()
    
    def plot_roc_curve(self, save_path=None):
        """Plot ROC curve for classification"""
        if self.problem_type != 'classification' or not hasattr(self, 'probabilities'):
            print("⚠️ ROC curve only available for classification with probabilities")
            return
        
        fpr, tpr, _ = roc_curve(self.y_test, self.probabilities)
        auc = roc_auc_score(self.y_test, self.probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            print(f"✅ ROC curve saved to {save_path}")
        plt.show()
    
    def plot_feature_importance(self, feature_names=None, top_n=20, save_path=None):
        """Plot feature importance for tree-based models"""
        if not hasattr(self.model, 'feature_importances_'):
            print("⚠️ Model doesn't have feature_importances_ attribute")
            return
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        plt.figure(figsize=(10, 8))
        plt.title(f'Top {top_n} Feature Importance')
        plt.barh(range(top_n), importances[indices][::-1], color='skyblue')
        plt.yticks(range(top_n), [feature_names[i] for i in indices[::-1]])
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"✅ Feature importance plot saved to {save_path}")
        plt.show()
        
        return {feature_names[i]: importances[i] for i in indices}
    
    def shap_analysis(self, X_train=None, feature_names=None, sample_size=100):
        """Perform SHAP analysis for model interpretability"""
        if not hasattr(self.model, 'predict'):
            print("⚠️ Model doesn't support SHAP analysis")
            return
        
        try:
            # Create a SHAP explainer
            if self.problem_type == 'classification' and hasattr(self.model, 'predict_proba'):
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(self.X_test[:sample_size])
                
                # For classification, shap_values might be a list
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Use class 1 (Pass)
                
                # Summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, self.X_test[:sample_size], 
                                 feature_names=feature_names, show=False)
                plt.title('SHAP Feature Importance')
                plt.tight_layout()
                plt.show()
                print("✅ SHAP analysis completed")
            else:
                print("⚠️ SHAP analysis only for classification with predict_proba")
        except Exception as e:
            print(f"⚠️ SHAP analysis failed: {e}")
    
    def plot_residuals(self, save_path=None):
        """Plot residuals for regression models"""
        if self.problem_type != 'regression':
            print("⚠️ Residuals plot only for regression")
            return
        
        residuals = self.y_test - self.predictions
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(self.predictions, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Distribution of residuals
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"✅ Residuals plot saved to {save_path}")
        plt.show()