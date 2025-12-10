#!/usr/bin/env python3
"""
Log all AutoGluon models to MLflow
Run this to populate MLflow UI with all 13 model experiments
"""
import pandas as pd
import mlflow

print("="*60)
print("LOGGING AUTOGLUON MODELS TO MLFLOW")
print("="*60)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("credit_card_default_prediction")

leaderboard = pd.read_csv('models/leaderboard.csv')
print(f"\nFound {len(leaderboard)} models in leaderboard")

best_model_name = leaderboard.iloc[0]['model']
best_metrics = {
    'accuracy': 0.7940,
    'precision': 0.5351,
    'recall': 0.5222,
    'f1_score': 0.5286,
    'roc_auc': 0.7746
}

# Log each model as separate MLflow run
for idx, row in leaderboard.iterrows():
    model_name = row['model']
    
    print(f"\nLogging: {model_name}")
    
    with mlflow.start_run(run_name=f"AutoGluon_{model_name}"):

        mlflow.log_param("automl_framework", "AutoGluon")
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("stack_level", int(row['stack_level']))
        mlflow.log_param("time_limit_seconds", 300)
        mlflow.log_param("preset", "best_quality")
        mlflow.log_param("is_best_model", model_name == best_model_name)
        mlflow.log_param("fit_order", int(row['fit_order']))
        
        mlflow.log_metric("test_f1_score", float(row['score_test']))
        mlflow.log_metric("validation_f1_score", float(row['score_val']))
        mlflow.log_metric("fit_time_seconds", float(row['fit_time']))
        mlflow.log_metric("prediction_time_seconds", float(row['pred_time_test']))
        
        # For best model, log additional detailed metrics
        if model_name == best_model_name:
            mlflow.log_metric("test_accuracy", best_metrics['accuracy'])
            mlflow.log_metric("test_precision", best_metrics['precision'])
            mlflow.log_metric("test_recall", best_metrics['recall'])
            mlflow.log_metric("test_roc_auc", best_metrics['roc_auc'])
            
            mlflow.log_text(leaderboard.to_string(), "full_leaderboard.txt")

print("\n" + "="*60)
print(f"✓ Successfully logged {len(leaderboard)} models to MLflow!")
print("="*60)
print("\nView in MLflow UI:")
print("  mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5001")
print("\nYou should now see 13 AutoGluon runs in the MLflow UI!")


