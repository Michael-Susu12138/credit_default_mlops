"""
Configuration settings for the MLOps pipeline
"""
import os

class Config:
    """Configuration class for the pipeline"""
    

    DATASET_NAME = "scikit-learn/credit-card-clients"
    DATA_DIR = "data"
    TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
    TEST_FILE = os.path.join(DATA_DIR, "test.csv")
    TEST_SPLIT = 0.2
    RANDOM_STATE = 42
    

    MODEL_DIR = "models"
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model")
    TARGET_COLUMN = "default.payment.next.month"
    

    MLFLOW_TRACKING_URI = "file:./mlruns"
    MLFLOW_EXPERIMENT_NAME = "credit_card_default_prediction"
    

    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = 5000
    

    MONITORING_DIR = "monitoring"
    REPORTS_DIR = os.path.join(MONITORING_DIR, "reports")
    

    AUTOGLUON_TIME_LIMIT = 300  
    AUTOGLUON_PRESET = 'best_quality'  
    AUTOGLUON_EVAL_METRIC = 'f1'  # F1 score for binary classification
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.MONITORING_DIR, exist_ok=True)
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)

