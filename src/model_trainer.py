"""
Model Trainer - Handles AutoML training using AutoGluon
"""
import pandas as pd
import numpy as np
import mlflow
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import os
import shutil
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Class to train and evaluate models using AutoGluon AutoML"""
    
    def __init__(self, config):
        """
        Initialize ModelTrainer
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.predictor = None
        self.best_model_name = None
        self.best_metrics = None
        self.leaderboard = None
    
    def _evaluate_model(self, y_true, y_pred, y_pred_proba):
        """Calculate evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
        return metrics
    
    def train_automl(self, X_train, y_train, X_test, y_test):
        """Train models using AutoGluon AutoML"""
        print("\n" + "="*60)
        print("AUTOML TRAINING WITH AUTOGLUON")
        print("="*60)

        mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.config.MLFLOW_EXPERIMENT_NAME)
        
        train_data = X_train.copy()
        train_data[self.config.TARGET_COLUMN] = y_train.values
        
        test_data = X_test.copy()
        test_data[self.config.TARGET_COLUMN] = y_test.values
        
        ag_path = os.path.join(self.config.MODEL_DIR, 'autogluon_models')
        
        print(f"\nAutoGluon Configuration:")
        print(f"  - Target: {self.config.TARGET_COLUMN}")
        print(f"  - Training samples: {len(train_data)}")
        print(f"  - Test samples: {len(test_data)}")
        print(f"  - Time limit: 300 seconds")
        print(f"  - Eval metric: f1 (weighted)")
        print(f"  - Presets: best_quality")
        print(f"\nStarting AutoGluon training...")
        
        self.predictor = TabularPredictor(
            label=self.config.TARGET_COLUMN,
            path=ag_path,
            eval_metric='f1',  
            problem_type='binary'
        ).fit(
            train_data=train_data,
            time_limit=300,  
            presets='best_quality',
            verbosity=2
        )
        
        print("\n" + "="*60)
        print("AUTOGLUON TRAINING COMPLETE")
        print("="*60)
        
        self.leaderboard = self.predictor.leaderboard(test_data, silent=True)
        print("\nModel Leaderboard:")
        print(self.leaderboard[['model', 'score_val', 'score_test', 'pred_time_test', 'fit_time']].to_string())
        
        self.best_model_name = self.leaderboard.iloc[0]['model']
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"{'='*60}")

        y_pred = self.predictor.predict(test_data)
        y_pred_proba = self.predictor.predict_proba(test_data)
        
        if hasattr(y_pred_proba, 'values'):
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                y_pred_proba = y_pred_proba.iloc[:, 1].values  
            else:
                y_pred_proba = y_pred_proba.values.flatten()
        
        self.best_metrics = self._evaluate_model(y_test, y_pred, y_pred_proba)
        
        print("\nTest Metrics:")
        for metric, value in self.best_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\n" + "="*60)
        print("LOGGING ALL MODELS TO MLFLOW")
        print("="*60)
        
        for idx, row in self.leaderboard.iterrows():
            model_name = row['model']
            
            with mlflow.start_run(run_name=f"AutoGluon_{model_name}"):

                mlflow.log_param("automl_framework", "AutoGluon")
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("stack_level", row.get('stack_level', 'N/A'))
                mlflow.log_param("time_limit", 300)
                mlflow.log_param("preset", "best_quality")
                mlflow.log_param("is_best_model", model_name == self.best_model_name)
                
                mlflow.log_metric("test_f1_score", row['score_test'])
                mlflow.log_metric("val_f1_score", row['score_val'])
                mlflow.log_metric("fit_time_seconds", row['fit_time'])
                mlflow.log_metric("pred_time_test_seconds", row['pred_time_test'])
                
                if model_name == self.best_model_name:
                    for metric, value in self.best_metrics.items():
                        if metric != 'f1_score':  
                            mlflow.log_metric(f"test_{metric}", value)
                    

                    mlflow.log_text(self.leaderboard.to_string(), "full_leaderboard.txt")
        
        print(f"✓ Logged {len(self.leaderboard)} models to MLflow")
        print("\nAutoGluon model training completed successfully!")
        
        return self.predictor, self.best_model_name, self.best_metrics
    
    def save_best_model(self):
        """Save the AutoGluon model"""
        if self.predictor is None:
            raise ValueError("No model trained. Call train_automl() first.")
        
        self.config.create_directories()
        info_file = f"{self.config.BEST_MODEL_PATH}_info.txt"
        with open(info_file, 'w') as f:
            f.write(f"AutoML Framework: AutoGluon\n")
            f.write(f"Best Model: {self.best_model_name}\n")
            f.write(f"\nTest Metrics:\n")
            for metric, value in self.best_metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write(f"\nModel Path: {self.predictor.path}\n")
        
        leaderboard_file = os.path.join(self.config.MODEL_DIR, 'leaderboard.csv')
        self.leaderboard.to_csv(leaderboard_file, index=False)
        
        print(f"\nAutoGluon model info saved to: {info_file}")
        print(f"Leaderboard saved to: {leaderboard_file}")
        print(f"AutoGluon models directory: {self.predictor.path}")
        
        return True

