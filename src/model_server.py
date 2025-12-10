"""
Model Server - REST API for model inference
"""
import pandas as pd
from autogluon.tabular import TabularPredictor
from flask import Flask, request, jsonify
import os

class ModelServer:
    """Class to serve the model via REST API"""
    
    def __init__(self, config):
        """
        Initialize ModelServer
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model = None
        self.feature_names = None
        self.app = Flask(__name__)
        self._setup_routes()
        
    def load_model(self):
        """Load the trained AutoGluon model"""
        ag_path = os.path.join(self.config.MODEL_DIR, 'autogluon_models')
        
        print(f"Loading AutoGluon model from: {ag_path}")
        
        if not os.path.exists(ag_path):
            print("AutoGluon model not found, checking for legacy model...")
            try:
                import mlflow.sklearn
                self.model = mlflow.sklearn.load_model(self.config.BEST_MODEL_PATH)
                print("Loaded legacy sklearn model")
            except Exception as e:
                raise FileNotFoundError(f"No model found. Please train the model first. Error: {e}")
        else:
            self.model = TabularPredictor.load(ag_path)
            print(f"AutoGluon model loaded successfully!")
            print(f"Best model: {self.model.model_best}")
        

        feature_file = os.path.join(self.config.DATA_DIR, 'feature_names.txt')
        with open(feature_file, 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        
        print(f"Features: {len(self.feature_names)}")
        
        return True
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'model_loaded': self.model is not None
            })
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """Prediction endpoint"""
            try:
                data = request.get_json()
                

                if 'instances' in data:
                    instances = data['instances']
                elif 'features' in data:
                    instances = [data['features']]
                else:
                    instances = [data]
                

                input_df = pd.DataFrame(instances)
                
                if self.feature_names:
                    input_df = input_df[self.feature_names]
                
                predictions = self.model.predict(input_df)
                probabilities = self.model.predict_proba(input_df)
                
                if hasattr(probabilities, 'values'):
                    proba_array = probabilities.values
                    if len(proba_array.shape) == 1:
                        proba_array = [[1-p, p] for p in proba_array]
                else:
                    proba_array = probabilities
                
                results = []
                for i in range(len(predictions)):
                    pred_val = int(predictions.iloc[i]) if hasattr(predictions, 'iloc') else int(predictions[i])
                    
                    if len(proba_array.shape) > 1:
                        prob_0 = float(proba_array[i][0])
                        prob_1 = float(proba_array[i][1])
                    else:
                        prob_1 = float(proba_array[i])
                        prob_0 = 1.0 - prob_1
                    
                    results.append({
                        'prediction': pred_val,
                        'probability_class_0': prob_0,
                        'probability_class_1': prob_1,
                        'predicted_class': 'No Default' if pred_val == 0 else 'Default'
                    })
                
                return jsonify({
                    'predictions': results,
                    'count': len(results)
                })
            
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        @self.app.route('/model_info', methods=['GET'])
        def model_info():
            """Get model information"""
            try:
                info_file = f"{self.config.BEST_MODEL_PATH}_info.txt"
                with open(info_file, 'r') as f:
                    info = f.read()
                
                return jsonify({
                    'model_info': info,
                    'features': self.feature_names,
                    'feature_count': len(self.feature_names)
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 400
    
    def run(self):
        """Start the Flask server"""
        if self.model is None:
            self.load_model()
        
        print("\n" + "="*60)
        print("MODEL SERVER STARTING")
        print("="*60)
        print(f"Server: http://{self.config.SERVER_HOST}:{self.config.SERVER_PORT}")
        print("\nEndpoints:")
        print("  - GET  /health       : Health check")
        print("  - POST /predict      : Make predictions")
        print("  - GET  /model_info   : Model information")
        print("="*60 + "\n")
        
        self.app.run(
            host=self.config.SERVER_HOST,
            port=self.config.SERVER_PORT,
            debug=False,
            use_reloader=False
        )

