"""
Monitoring Manager - Handles model monitoring with Evidently
"""
import pandas as pd
import numpy as np
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    EVIDENTLY_NEW_API = True
except ImportError:
    try:
        from evidently import Report
        from evidently.metrics import DataDriftTable, DatasetDriftMetric
        EVIDENTLY_NEW_API = False
    except ImportError:
        Report = None
        EVIDENTLY_NEW_API = None

import os
import json
from datetime import datetime
from scipy import stats

class MonitoringManager:
    """Class to handle model monitoring"""
    
    def __init__(self, config):
        """
        Initialize MonitoringManager
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.target_column = None
        self.prediction_column = None
        self.reference_data = None
        self.current_data = None
        
    def setup_column_mapping(self, target_column, prediction_column='prediction'):
        """Setup column mapping for Evidently"""
        self.target_column = target_column
        self.prediction_column = prediction_column
        
        return True
    
    def set_reference_data(self, reference_df):
        """Set reference data (typically training data)"""
        self.reference_data = reference_df.copy()
        print(f"Reference data set: {self.reference_data.shape}")
    
    def generate_data_drift_report(self, current_df, report_name='data_drift'):
        """Generate data drift report"""
        print(f"\n{'='*60}")
        print("GENERATING DATA DRIFT REPORT")
        print(f"{'='*60}")
        
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")
        
        self.current_data = current_df.copy()
        
        drift_info = self._detect_drift_simple()
        
        self.config.create_directories()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = os.path.join(
            self.config.REPORTS_DIR,
            f'{report_name}_{timestamp}_metrics.json'
        )
        with open(metrics_path, 'w') as f:
            json.dump(drift_info, f, indent=2)
        
        print(f"Drift metrics saved: {metrics_path}")
        
        report_path = os.path.join(
            self.config.REPORTS_DIR,
            f'{report_name}_{timestamp}.html'
        )
        self._create_html_report(drift_info, report_path)
        print(f"HTML report saved: {report_path}")
        
        return None, drift_info
    
    def _detect_drift_simple(self):
        """Simple drift detection using Kolmogorov-Smirnov test"""
        drift_info = {
            'timestamp': datetime.now().isoformat(),
            'reference_size': len(self.reference_data),
            'current_size': len(self.current_data),
            'drift_detected': False,
            'drifted_features': [],
            'drift_scores': {}
        }
        
        exclude_cols = [self.target_column, self.prediction_column]
        numerical_cols = [col for col in self.reference_data.columns 
                         if col not in exclude_cols and 
                         pd.api.types.is_numeric_dtype(self.reference_data[col])]
        
        drifted_count = 0
        for col in numerical_cols:
            if col in self.current_data.columns:
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(
                    self.reference_data[col].dropna(),
                    self.current_data[col].dropna()
                )
                
                drift_info['drift_scores'][col] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value)
                }
                
                if p_value < 0.05:
                    drift_info['drifted_features'].append(col)
                    drifted_count += 1
        
        drift_info['drift_detected'] = drifted_count > 0
        drift_info['number_of_drifted_columns'] = drifted_count
        drift_info['drift_share'] = drifted_count / max(len(numerical_cols), 1)
        
        return drift_info
    
    def _create_html_report(self, drift_info, report_path):
        """Create a simple HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Drift Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                .drift {{ background-color: #ffcccc; }}
                .no-drift {{ background-color: #ccffcc; }}
            </style>
        </head>
        <body>
            <h1>Data Drift Report</h1>
            <p><strong>Timestamp:</strong> {drift_info['timestamp']}</p>
            <p><strong>Reference Data Size:</strong> {drift_info['reference_size']}</p>
            <p><strong>Current Data Size:</strong> {drift_info['current_size']}</p>
            <p><strong>Drift Detected:</strong> {drift_info['drift_detected']}</p>
            <p><strong>Number of Drifted Features:</strong> {drift_info['number_of_drifted_columns']}</p>
            <p><strong>Drift Share:</strong> {drift_info['drift_share']:.2%}</p>
            
            <h2>Feature Drift Details</h2>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>KS Statistic</th>
                    <th>P-Value</th>
                    <th>Drift Status</th>
                </tr>
        """
        
        for feature, scores in drift_info['drift_scores'].items():
            is_drifted = feature in drift_info['drifted_features']
            row_class = 'drift' if is_drifted else 'no-drift'
            status = 'DRIFT DETECTED' if is_drifted else 'No Drift'
            
            html_content += f"""
                <tr class="{row_class}">
                    <td>{feature}</td>
                    <td>{scores['statistic']:.4f}</td>
                    <td>{scores['p_value']:.4f}</td>
                    <td>{status}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
    
    def _extract_drift_info(self, report_dict):
        """Extract drift information from report"""
        drift_info = {
            'timestamp': datetime.now().isoformat(),
            'reference_size': len(self.reference_data),
            'current_size': len(self.current_data),
            'drift_detected': False,
            'drifted_features': [],
            'drift_scores': {}
        }
        
        try:
            for metric in report_dict.get('metrics', []):
                if metric.get('metric') == 'DatasetDriftMetric':
                    result = metric.get('result', {})
                    drift_info['drift_detected'] = result.get('dataset_drift', False)
                    drift_info['drift_share'] = result.get('drift_share', 0)
                    drift_info['number_of_drifted_columns'] = result.get('number_of_drifted_columns', 0)
                    
                    drift_by_columns = result.get('drift_by_columns', {})
                    for col, col_info in drift_by_columns.items():
                        if col_info.get('drift_detected', False):
                            drift_info['drifted_features'].append(col)
                            drift_info['drift_scores'][col] = col_info.get('drift_score', 0)
        
        except Exception as e:
            print(f"Warning: Could not extract full drift info: {e}")
        
        return drift_info
    
    def generate_model_performance_report(self, predictions_df, report_name='model_performance'):
        """Generate model performance report"""
        print(f"\n{'='*60}")
        print("GENERATING MODEL PERFORMANCE REPORT")
        print(f"{'='*60}")
        
        if self.reference_data is None:
            raise ValueError("Reference data not set.")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(
            self.config.REPORTS_DIR,
            f'{report_name}_{timestamp}.html'
        )
        
        print(f"Performance report would be saved to: {report_path}")
        print("(Simplified monitoring - full Evidently integration available with compatible version)")
        
        return None, report_path
    
    def print_drift_summary(self, drift_info):
        """Print a summary of drift detection"""
        print(f"\n{'='*60}")
        print("DRIFT DETECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Dataset Drift Detected: {drift_info['drift_detected']}")
        print(f"Number of Drifted Features: {len(drift_info['drifted_features'])}")
        
        if drift_info['drifted_features']:
            print(f"\nDrifted Features:")
            for feature in drift_info['drifted_features']:
                score = drift_info['drift_scores'].get(feature, 'N/A')
                print(f"  - {feature}: {score}")
        else:
            print("\nNo feature drift detected.")
        
        print(f"{'='*60}")

