"""
Test script for the enhanced anomaly detection functionality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from tools.predictive_analytics import PredictiveAnalyticsTools

def main():
    print("Testing enhanced anomaly detection...")
    
    # Test with synthetic data first
    print("\n1. Testing with synthetic data containing known anomalies:")
    # Create synthetic time series with anomalies
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    values = np.sin(np.linspace(0, 4*np.pi, 100)) * 10 + 50  # Sine wave
    
    # Add trend
    values = values + np.linspace(0, 10, 100)
    
    # Add noise
    values = values + np.random.normal(0, 1, 100)
    
    # Add specific anomalies
    values[20] += 15  # Single point anomaly
    values[50:53] -= 20  # Multiple point anomaly
    values[80] += 25  # Large anomaly
    
    # Create series
    synthetic_series = pd.Series(values, index=dates)
    
    # Run anomaly detection with multiple methods
    results_synthetic = PredictiveAnalyticsTools.detect_anomalies(
        synthetic_series,
        methods=['isolation_forest', 'lof', 'zscore', 'iqr', 'rolling_zscore'],
        window_size=10,
        zscore_threshold=2.5
    )
    
    # Print results
    print(f"Methods used: {results_synthetic['methods_used']}")
    print("\nResults by method:")
    for method, result in results_synthetic['method_results'].items():
        if 'error' not in result:
            print(f"- {method}: {result['anomaly_count']} anomalies ({result['anomaly_percent']:.2f}%)")
    
    if 'ensemble_results' in results_synthetic and results_synthetic['ensemble_results']:
        print(f"\nEnsemble results: {results_synthetic['ensemble_results']['anomaly_count']} anomalies")
        print(f"Ensemble anomaly indices: {results_synthetic['ensemble_results']['anomaly_indices']}")
    
    print(f"\nPlot saved to: {results_synthetic['anomaly_plot_path']}")
    if results_synthetic.get('anomaly_table_path'):
        print(f"Table saved to: {results_synthetic['anomaly_table_path']}")
    
    # Test with real stock data
    print("\n\n2. Testing with real stock data (AAPL):")
    try:
        # Download Apple stock data for the past year
        aapl_data = yf.download('AAPL', start='2023-01-01', end='2023-12-31')
        print(f"Downloaded {len(aapl_data)} days of AAPL data")
        
        if len(aapl_data) == 0:
            print("No data downloaded for AAPL")
        else:
            print(f"AAPL data columns: {aapl_data.columns.tolist()}")
            print(f"AAPL data sample:\n{aapl_data.head()}")
            
            # Extract the Close price series properly
            print("\nExtracting AAPL closing prices...")
            # Handle multi-level columns if present
            if isinstance(aapl_data.columns, pd.MultiIndex):
                close_series = aapl_data[('Close', 'AAPL')]
                print("Using multi-index column ('Close', 'AAPL')")
            else:
                close_series = aapl_data['Close']
                print("Using standard column 'Close'")
                
            print(f"Close series type: {type(close_series)}")
            print(f"Close series sample:\n{close_series.head()}")
            
            # Run anomaly detection on closing prices
            print("\nRunning anomaly detection on AAPL closing prices...")
            results_aapl = PredictiveAnalyticsTools.detect_anomalies(
                close_series,
                methods=['isolation_forest', 'lof', 'zscore'],
                contamination=0.03
            )
            
            # Debug the results
            print(f"\nKeys in results_aapl: {list(results_aapl.keys())}")
            
            if 'error' in results_aapl:
                print(f"Error in anomaly detection: {results_aapl['error']}")
            else:
                # Print results
                print(f"Methods used: {results_aapl.get('methods_used', [])}")
                print("\nResults by method:")
                for method, result in results_aapl.get('method_results', {}).items():
                    if 'error' not in result:
                        print(f"- {method}: {result['anomaly_count']} anomalies ({result['anomaly_percent']:.2f}%)")
                
                if 'ensemble_results' in results_aapl and results_aapl['ensemble_results']:
                    print(f"\nEnsemble results: {results_aapl['ensemble_results']['anomaly_count']} anomalies")
                
                print(f"\nPlot saved to: {results_aapl.get('anomaly_plot_path', 'N/A')}")
                if results_aapl.get('anomaly_table_path'):
                    print(f"Table saved to: {results_aapl['anomaly_table_path']}")
    
    except Exception as e:
        import traceback
        print(f"Error testing with real stock data: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
