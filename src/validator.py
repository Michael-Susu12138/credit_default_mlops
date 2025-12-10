"""
Model Validator - Handles model testing and validation
"""
import pandas as pd
import numpy as np
import requests
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
import time

class ModelValidator:
    """Class to validate deployed model"""
    
    def __init__(self, config):
        """
        Initialize ModelValidator
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.server_url = f"http://localhost:{config.SERVER_PORT}"
        
    def wait_for_server(self, timeout=60, retry_interval=2):
        """Wait for server to be ready"""
        print(f"Waiting for server at {self.server_url}...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.server_url}/health", timeout=5)
                if response.status_code == 200:
                    print("Server is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(retry_interval)
        
        raise TimeoutError(f"Server not ready after {timeout} seconds")
    
    def predict(self, X):
        """Make predictions using deployed model"""
        instances = X.to_dict(orient='records')
        
        print(f"Making prediction request to {self.server_url}/predict")
        print(f"Sending {len(instances)} instances")
        
        try:
            response = requests.post(
                f"{self.server_url}/predict",
                json={'instances': instances},
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"ERROR: Status code {response.status_code}")
                print(f"Response: {response.text[:500]}")
                raise Exception(f"Prediction failed (status {response.status_code}): {response.text[:200]}")
            
            result = response.json()
            predictions = result['predictions']
            y_pred = [p['prediction'] for p in predictions]
            y_proba = [p['probability_class_1'] for p in predictions]
            
            return np.array(y_pred), np.array(y_proba)
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Request failed - {e}")
            raise
    
    def evaluate(self, X, y_true, dataset_name='Test'):
        """Evaluate model on dataset"""
        print(f"\n{'='*60}")
        print(f"EVALUATING ON {dataset_name.upper()} DATA")
        print(f"{'='*60}")
        

        y_pred, y_proba = self.predict(X)
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba)
        }
        
        print(f"\nMetrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        return metrics, y_pred, y_proba
    
    def create_modified_data(self, X, n_features_to_modify=2):
        """Create modified test data by randomly changing feature values"""
        print(f"\n{'='*60}")
        print(f"MODIFYING TEST DATA")
        print(f"{'='*60}")
        
        X_modified = X.copy()
        
        features = X.columns.tolist()
        np.random.seed(42)
        features_to_modify = np.random.choice(features, n_features_to_modify, replace=False)
        
        print(f"\nFeatures being modified: {features_to_modify.tolist()}")
        
        for feature in features_to_modify:
            original_values = X_modified[feature].values
            
            shuffled_indices = np.random.permutation(len(original_values))
            X_modified[feature] = original_values[shuffled_indices]
            
            print(f"  - {feature}: values shuffled")
        
        print(f"\nOriginal data sample:")
        print(X.head())
        print(f"\nModified data sample:")
        print(X_modified.head())
        
        return X_modified, features_to_modify.tolist()
    
    def compare_predictions(self, original_metrics, modified_metrics):
        """Compare predictions between original and modified data"""
        print(f"\n{'='*60}")
        print("METRICS COMPARISON")
        print(f"{'='*60}")
        
        comparison = pd.DataFrame({
            'Original': original_metrics,
            'Modified': modified_metrics,
            'Difference': {k: modified_metrics[k] - original_metrics[k] 
                          for k in original_metrics.keys()}
        })
        
        print(comparison.to_string())
        
        return comparison

