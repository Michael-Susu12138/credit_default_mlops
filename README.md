# MLOps Pipeline - Credit Card Default Prediction

A complete end-to-end MLOps pipeline for predicting credit card defaults using the UCI ML credit card dataset.

## Features

- **Object-Oriented Design**: Clean class-based architecture
- **AutoML Training**: Uses **AutoGluon** - professional AutoML framework
- **Model Deployment**: REST API for inference
- **Model Monitoring**: Drift detection with statistical tests
- **MLflow Integration**: Experiment tracking and model versioning

## Project Structure

```
mlops_final/
├── config.py              # Configuration settings
├── main.py                # Main pipeline orchestrator
├── requirements.txt       # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_manager.py    # Data loading and preparation
│   ├── model_trainer.py   # AutoML training
│   ├── model_server.py    # REST API server
│   ├── monitoring.py      # Model monitoring
│   └── validator.py       # Model validation
├── data/                  # Data files (generated)
├── models/                # Trained models (generated)
├── mlruns/                # MLflow tracking (generated)
└── monitoring/            # Monitoring reports (generated)
```

## Setup

1. **Activate virtual environment**:
```bash
source /home/ubuntu/venv/bin/activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Run Complete Pipeline

Execute the entire MLOps pipeline with one command:

```bash
python main.py
```

This will:
1. Load and prepare the credit card default dataset
2. Train multiple models using AutoML
3. Select and save the best model
4. Deploy the model as a REST API
5. Set up monitoring with Evidently
6. Validate with original test data
7. Test with modified data and verify drift detection

### Individual Components

You can also use individual components:

```python
from config import Config
from src.data_manager import DataManager
from src.model_trainer import ModelTrainer

# Data preparation
config = Config()
data_manager = DataManager(config)
train_df, test_df, features = data_manager.prepare_data()
data_manager.save_data()

# Model training
trainer = ModelTrainer(config)
X_train, y_train = data_manager.get_X_y(train_df)
X_test, y_test = data_manager.get_X_y(test_df)
model, name, metrics = trainer.train_automl(X_train, y_train, X_test, y_test)
trainer.save_best_model()
```

## API Endpoints

When the model server is running:

- `GET /health` - Health check
- `POST /predict` - Make predictions
- `GET /model_info` - Get model information

Example prediction request:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"LIMIT_BAL": 20000, "SEX": 2, ...}]}'
```

## Monitoring

The pipeline generates HTML reports in `monitoring/reports/`:
- Data drift reports
- Model performance reports
- Drift detection metrics

View MLflow UI:
```bash
mlflow ui --backend-store-uri file:./mlruns
```

## AutoML with AutoGluon

The pipeline uses **AutoGluon** which automatically evaluates:
- LightGBM, XGBoost, CatBoost
- Random Forest, Extra Trees
- K-Nearest Neighbors
- Neural Networks (PyTorch & FastAI)
- Weighted Ensemble models (stacking/blending)

AutoGluon automatically:
- Tunes hyperparameters
- Creates ensemble models
- Selects the best model based on F1 score
- Handles feature engineering

**Typical result:** 10-20+ models trained, best is usually a weighted ensemble.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Configuration

Modify `config.py` to customize:
- Data split ratios
- Model hyperparameters
- Server settings
- Monitoring thresholds

