"""
Data Manager - Handles data loading, preparation, and splitting
"""
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import os

class DataManager:
    """Class to manage data operations"""
    
    def __init__(self, config):
        """
        Initialize DataManager
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.train_df = None
        self.test_df = None
        self.feature_names = None
        
    def load_dataset(self):
        """Load dataset from HuggingFace"""
        print(f"Loading dataset from HuggingFace: {self.config.DATASET_NAME}")
        
        dataset = load_dataset(self.config.DATASET_NAME)
        
        df = dataset['train'].to_pandas()
        
        print(f"✓ Dataset loaded successfully!")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Rename target column if needed to match our config
        # The HuggingFace dataset uses 'default.payment.next.month'
        if 'default.payment.next.month' in df.columns and self.config.TARGET_COLUMN != 'default.payment.next.month':
            df.rename(columns={'default.payment.next.month': self.config.TARGET_COLUMN}, inplace=True)
        elif 'default payment next month' in df.columns:
            df.rename(columns={'default payment next month': self.config.TARGET_COLUMN}, inplace=True)

        if self.config.TARGET_COLUMN not in df.columns:
            # Find any column with 'default' in the name
            default_cols = [col for col in df.columns if 'default' in col.lower()]
            if default_cols:
                print(f"  Renaming '{default_cols[0]}' to '{self.config.TARGET_COLUMN}'")
                df.rename(columns={default_cols[0]: self.config.TARGET_COLUMN}, inplace=True)
            else:
                raise ValueError(f"Target column '{self.config.TARGET_COLUMN}' not found in dataset")
        
        print(f"  Target variable: {self.config.TARGET_COLUMN}")
        print(f"  Target distribution:\n{df[self.config.TARGET_COLUMN].value_counts()}")
        
        return df
    
    def prepare_data(self):
        """Prepare and split data into train/test sets"""
        print("\n" + "="*60)
        print("DATA PREPARATION")
        print("="*60)
        

        df = self.load_dataset()
        
        X = df.drop(self.config.TARGET_COLUMN, axis=1)
        y = df[self.config.TARGET_COLUMN]
        
        self.feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.TEST_SPLIT,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )
        
        print(f"\nTrain set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        print(f"Number of features: {len(self.feature_names)}")
        
        self.train_df = pd.concat([X_train, y_train], axis=1)
        self.test_df = pd.concat([X_test, y_test], axis=1)
        
        return self.train_df, self.test_df, self.feature_names
    
    def save_data(self):
        """Save prepared data to disk"""
        if self.train_df is None or self.test_df is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        self.config.create_directories()
        
        self.train_df.to_csv(self.config.TRAIN_FILE, index=False)
        self.test_df.to_csv(self.config.TEST_FILE, index=False)
        
        feature_file = os.path.join(self.config.DATA_DIR, 'feature_names.txt')
        with open(feature_file, 'w') as f:
            f.write('\n'.join(self.feature_names))
        
        print(f"\nData saved:")
        print(f"  - {self.config.TRAIN_FILE}")
        print(f"  - {self.config.TEST_FILE}")
        print(f"  - {feature_file}")
        
        return True
    
    def load_prepared_data(self):
        """Load previously prepared data"""
        self.train_df = pd.read_csv(self.config.TRAIN_FILE)
        self.test_df = pd.read_csv(self.config.TEST_FILE)
        
        feature_file = os.path.join(self.config.DATA_DIR, 'feature_names.txt')
        with open(feature_file, 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        
        return self.train_df, self.test_df, self.feature_names
    
    def get_X_y(self, df=None):
        """Split DataFrame into features and target"""
        if df is None:
            df = self.train_df
        
        X = df.drop(self.config.TARGET_COLUMN, axis=1)
        y = df[self.config.TARGET_COLUMN]
        
        return X, y

