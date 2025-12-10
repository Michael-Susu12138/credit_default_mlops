#!/usr/bin/env python3
"""
Main MLOps Pipeline Orchestrator
Runs the complete pipeline: data preparation, training, deployment, and monitoring
"""
import sys
import os
import subprocess
import time
import signal

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from src.data_manager import DataManager
from src.model_trainer import ModelTrainer
from src.model_server import ModelServer
from src.monitoring import MonitoringManager
from src.validator import ModelValidator

class MLOpsPipeline:
    """Main orchestrator for the MLOps pipeline"""
    
    def __init__(self):
        """Initialize the pipeline"""
        self.config = Config()
        self.data_manager = DataManager(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.model_server = ModelServer(self.config)
        self.monitoring = MonitoringManager(self.config)
        self.validator = ModelValidator(self.config)
        self.server_process = None
        
    def run_full_pipeline(self):
        """Execute the complete MLOps pipeline"""
        print("\n" + "="*60)
        print("MLOPS PIPELINE - CREDIT CARD DEFAULT PREDICTION")
        print("="*60)
        
        try:
            # Check if model already exists
            model_exists = self._check_model_exists()
            
            if model_exists:
                print("\n✓ MODEL ALREADY TRAINED - SKIPPING TRAINING")
                print("  Loading existing model and data...")
                
                # Load existing data
                if not os.path.exists(self.config.TRAIN_FILE):
                    print("\n[STEP 1] Preparing data...")
                    train_df, test_df, feature_names = self.data_manager.prepare_data()
                    self.data_manager.save_data()
                else:
                    print("\n[STEP 1] Loading existing data...")
                    train_df, test_df, feature_names = self.data_manager.load_prepared_data()
                
                print("[STEP 2] Model training - SKIPPED (using existing model)")
            else:
                print("\n✗ NO TRAINED MODEL FOUND - STARTING TRAINING")
                print("  This will take approximately 5 minutes...")
                
                # Step 1: Data Preparation
                print("\n[STEP 1] Data Preparation")
                train_df, test_df, feature_names = self.data_manager.prepare_data()
                self.data_manager.save_data()
                
                # Step 2: Model Training with AutoGluon AutoML
                print("\n[STEP 2] Model Training with AutoGluon AutoML")
                X_train, y_train = self.data_manager.get_X_y(train_df)
                X_test, y_test = self.data_manager.get_X_y(test_df)
                
                predictor, best_model_name, best_metrics = self.model_trainer.train_automl(
                    X_train, y_train, X_test, y_test
                )
                self.model_trainer.save_best_model()
            
            # Continue with deployment and testing
            X_test, y_test = self.data_manager.get_X_y(test_df)
            
            # Step 3: Deploy Model
            print("\n[STEP 3] Deploying Model")
            self.server_process = self._start_server_background()
            
            # Wait for server to start
            self.validator.wait_for_server()
            
            # Step 4: Set up Monitoring
            print("\n[STEP 4] Setting Up Monitoring")
            self.monitoring.setup_column_mapping(
                target_column=self.config.TARGET_COLUMN,
                prediction_column='prediction'
            )
            
            # Use training data as reference
            train_sample = train_df.sample(min(1000, len(train_df)), random_state=42)
            self.monitoring.set_reference_data(train_sample)
            
            # Step 5: Validate with Original Test Data
            print("\n[STEP 5] Validating with Original Test Data")
            original_metrics, y_pred_orig, y_proba_orig = self.validator.evaluate(
                X_test, y_test, dataset_name='Original Test'
            )
            
            # Create predictions DataFrame for monitoring
            test_with_preds = test_df.copy()
            test_with_preds['prediction'] = y_pred_orig
            
            # Generate monitoring report for original data
            report_orig, drift_info_orig = self.monitoring.generate_data_drift_report(
                test_with_preds, report_name='original_test_data'
            )
            self.monitoring.print_drift_summary(drift_info_orig)
            
            # Step 6: Test with Modified Data
            print("\n[STEP 6] Testing with Modified Data")
            X_test_modified, modified_features = self.validator.create_modified_data(
                X_test, n_features_to_modify=2
            )
            
            modified_metrics, y_pred_mod, y_proba_mod = self.validator.evaluate(
                X_test_modified, y_test, dataset_name='Modified Test'
            )
            
            # Create predictions DataFrame for modified data
            test_modified_df = test_df.copy()
            for feature in modified_features:
                test_modified_df[feature] = X_test_modified[feature]
            test_modified_df['prediction'] = y_pred_mod
            
            # Generate monitoring report for modified data
            report_mod, drift_info_mod = self.monitoring.generate_data_drift_report(
                test_modified_df, report_name='modified_test_data'
            )
            self.monitoring.print_drift_summary(drift_info_mod)
            
            # Step 7: Compare Results
            print("\n[STEP 7] Comparing Results")
            comparison = self.validator.compare_predictions(original_metrics, modified_metrics)
            
            # Final Summary
            print("\n" + "="*60)
            print("PIPELINE EXECUTION COMPLETE")
            print("="*60)
            
            # Get best model name from saved info
            best_model_name = self._get_best_model_name()
            
            print(f"\nBest Model: {best_model_name}")
            print(f"Original Test F1 Score: {original_metrics['f1_score']:.4f}")
            print(f"Modified Test F1 Score: {modified_metrics['f1_score']:.4f}")
            print(f"Performance Drop: {(original_metrics['f1_score'] - modified_metrics['f1_score']):.4f}")
            print(f"\nOriginal Data Drift: {drift_info_orig['drift_detected']}")
            print(f"Modified Data Drift: {drift_info_mod['drift_detected']}")
            print(f"Modified Features: {modified_features}")
            
            if drift_info_mod['drift_detected']:
                print(f"\n✓ Drift detection working! Detected drift in modified data.")
                print(f"  Drifted features: {drift_info_mod['drifted_features']}")
            
            print(f"\nMonitoring Reports:")
            print(f"  - Check {self.config.REPORTS_DIR}/ for HTML reports")
            print(f"  - MLflow UI: mlflow ui --backend-store-uri {self.config.MLFLOW_TRACKING_URI}")
            print(f"\n✓ All pipeline steps completed successfully!")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"\nError in pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # Cleanup
            self._stop_server()
    
    def _check_model_exists(self):
        """Check if a trained model already exists"""
        autogluon_path = os.path.join(self.config.MODEL_DIR, 'autogluon_models', 'predictor.pkl')
        model_info_path = self.config.BEST_MODEL_PATH + '_info.txt'
        
        return os.path.exists(autogluon_path) or os.path.exists(model_info_path)
    
    def _get_best_model_name(self):
        """Get the best model name from saved files"""
        try:
            info_file = self.config.BEST_MODEL_PATH + '_info.txt'
            if os.path.exists(info_file):
                with open(info_file, 'r') as f:
                    for line in f:
                        if line.startswith('Best Model:'):
                            return line.split(':', 1)[1].strip()
            return "AutoGluon Ensemble"
        except:
            return "AutoGluon Model"
    
    def _start_server_background(self):
        """Start the model server in background"""
        print("Starting model server in background...")
        
        # Kill any existing server on the port
        try:
            import subprocess
            subprocess.run(
                f"lsof -ti:{self.config.SERVER_PORT} | xargs kill -9",
                shell=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL
            )
            time.sleep(2)
        except:
            pass
        
        # Create a simple server script
        server_script = os.path.join(os.path.dirname(__file__), '_temp_server.py')
        with open(server_script, 'w') as f:
            f.write('''
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/ubuntu/mlops_final')
os.chdir('/home/ubuntu/mlops_final')

from config import Config
from src.model_server import ModelServer

print("Initializing server...")
config = Config()
server = ModelServer(config)
print("Loading model...")
server.load_model()
print("Starting Flask server...")
server.run()
''')
        
        # Create log file for debugging
        log_file = os.path.join(os.path.dirname(__file__), 'server.log')
        
        # Start server in subprocess with output logging
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                [sys.executable, server_script],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(__file__)
            )
        
        time.sleep(15)  # Give server more time to load large AutoGluon model
        
        # Check if process is still alive
        if process.poll() is not None:
            print("ERROR: Server process died! Check server.log for details")
            with open(log_file, 'r') as log:
                print(log.read())
            raise RuntimeError("Server failed to start")
        
        return process
    
    def _stop_server(self):
        """Stop the model server"""
        if self.server_process:
            print("\nStopping model server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            print("Model server stopped.")

def main():
    """Main entry point"""
    pipeline = MLOpsPipeline()
    
    try:
        success = pipeline.run_full_pipeline()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        pipeline._stop_server()
        sys.exit(1)

if __name__ == "__main__":
    main()

