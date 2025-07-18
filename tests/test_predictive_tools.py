"""
Comprehensive test script for enhanced predictive analytics tools
Tests anomaly detection, volatility analysis, and scenario analysis with real market data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import os
from tools.predictive_analytics import PredictiveAnalyticsTools

def test_anomaly_detection(symbol='AAPL', period='1y'):
    """Test the enhanced anomaly detection functionality"""
    print("\n" + "="*80)
    print(f"TESTING ANOMALY DETECTION FOR {symbol}")
    print("="*80)
    
    try:
        # Download data
        data = yf.download(symbol, period=period)
        print(f"Downloaded {len(data)} days of {symbol} data")
        
        # Extract close prices
        if isinstance(data.columns, pd.MultiIndex):
            close_series = data[('Close', symbol)]
        else:
            close_series = data['Close']
        
        # Run anomaly detection with multiple methods
        print("\nRunning anomaly detection with multiple methods...")
        results = PredictiveAnalyticsTools.detect_anomalies(
            close_series,
            methods=['isolation_forest', 'lof', 'zscore', 'rolling_zscore'],
            contamination=0.03,
            window_size=20,
            zscore_threshold=2.5
        )
        
        # Print results
        if 'error' in results:
            print(f"Error in anomaly detection: {results['error']}")
            return
        
        print(f"\nMethods used: {results['methods_used']}")
        print("\nResults by method:")
        for method, result in results['method_results'].items():
            if 'error' not in result:
                print(f"- {method}: {result['anomaly_count']} anomalies ({result['anomaly_percent']:.2f}%)")
                if result['anomaly_count'] > 0:
                    print(f"  First 3 anomaly dates: {result['anomaly_indices'][:3]}")
        
        if 'ensemble_results' in results and results['ensemble_results']:
            print(f"\nEnsemble results: {results['ensemble_results']['anomaly_count']} anomalies")
            if results['ensemble_results']['anomaly_count'] > 0:
                print(f"First 3 ensemble anomaly dates: {results['ensemble_results']['anomaly_indices'][:3]}")
        
        print(f"\nPlot saved to: {results['anomaly_plot_path']}")
        if results.get('anomaly_table_path'):
            print(f"Table saved to: {results['anomaly_table_path']}")
        
        return results
    
    except Exception as e:
        import traceback
        print(f"Error in anomaly detection test: {str(e)}")
        traceback.print_exc()
        return None

def test_volatility_analysis(symbol='AAPL', period='1y'):
    """Test the enhanced volatility analysis functionality"""
    print("\n" + "="*80)
    print(f"TESTING VOLATILITY ANALYSIS FOR {symbol}")
    print("="*80)
    
    try:
        # Download data
        data = yf.download(symbol, period=period)
        print(f"Downloaded {len(data)} days of {symbol} data")
        
        # Run volatility analysis
        print("\nRunning volatility analysis...")
        results = PredictiveAnalyticsTools.calculate_volatility(
            data,
            price_column='Close' if not isinstance(data.columns, pd.MultiIndex) else ('Close', symbol),
            window_size=20,
            forecast_periods=30,
            use_garch=True,
            confidence_level=0.95
        )
        
        # Print results
        if 'error' in results:
            print(f"Error in volatility analysis: {results['error']}")
            return
        
        print("\nVolatility Statistics:")
        print(f"- Current volatility: {results.get('current_volatility', 'N/A'):.2%}")
        print(f"- Average volatility: {results.get('average_volatility', 'N/A'):.2%}")
        print(f"- Volatility regime: {results.get('volatility_regime', 'N/A')}")
        print(f"- Volatility trend: {results.get('volatility_trend', 'N/A')}")
        
        if 'var_95' in results:
            print(f"- Value at Risk (95%): {results['var_95']:.2%}")
        
        if 'volatility_forecast' in results:
            print("\nVolatility Forecast:")
            forecast = results['volatility_forecast']
            print(f"- Forecast periods: {len(forecast)}")
            print(f"- Last historical volatility: {forecast[0]['value']:.2%}")
            print(f"- Last forecast volatility: {forecast[-1]['value']:.2%}")
        
        if 'garch_summary' in results:
            print("\nGARCH Model Available: Yes")
        else:
            print("\nGARCH Model Available: No")
        
        print(f"\nVolatility plot saved to: {results.get('volatility_plot_path', 'N/A')}")
        if 'forecast_plot_path' in results:
            print(f"Forecast plot saved to: {results['forecast_plot_path']}")
        
        return results
    
    except Exception as e:
        import traceback
        print(f"Error in volatility analysis test: {str(e)}")
        traceback.print_exc()
        return None

def test_scenario_analysis(symbol='AAPL', period='1y'):
    """Test the enhanced scenario analysis functionality"""
    print("\n" + "="*80)
    print(f"TESTING SCENARIO ANALYSIS FOR {symbol}")
    print("="*80)
    
    try:
        # Download data
        data = yf.download(symbol, period=period)
        print(f"Downloaded {len(data)} days of {symbol} data")
        
        # Define custom scenarios
        custom_scenarios = {
            'optimistic': {'drift_factor': 1.5, 'volatility_factor': 0.8},
            'pessimistic': {'drift_factor': 0.5, 'volatility_factor': 1.5},
            'high_volatility': {'drift_factor': 1.0, 'volatility_factor': 2.0}
        }
        
        # Run scenario analysis
        print("\nRunning scenario analysis...")
        results = PredictiveAnalyticsTools.scenario_analysis(
            data,
            price_column='Close' if not isinstance(data.columns, pd.MultiIndex) else ('Close', symbol),
            scenarios=['base', 'bull', 'bear'],
            forecast_periods=60,
            simulations=300,
            custom_scenarios=custom_scenarios
        )
        
        # Print results
        if 'error' in results:
            print(f"Error in scenario analysis: {results['error']}")
            return
        
        print("\nScenario Analysis Results:")
        for scenario, stats in results.get('scenario_stats', {}).items():
            print(f"\n{scenario.upper()} SCENARIO:")
            print(f"- Mean final price: ${stats.get('mean_final_price', 'N/A'):.2f}")
            print(f"- Median final price: ${stats.get('median_final_price', 'N/A'):.2f}")
            print(f"- 95% CI: ${stats.get('ci_lower', 'N/A'):.2f} to ${stats.get('ci_upper', 'N/A'):.2f}")
            print(f"- Expected return: {stats.get('expected_return', 'N/A'):.2%}")
        
        print(f"\nScenario plot saved to: {results.get('scenario_plot_path', 'N/A')}")
        if 'scenario_table_path' in results:
            print(f"Scenario table saved to: {results['scenario_table_path']}")
        
        return results
    
    except Exception as e:
        import traceback
        print(f"Error in scenario analysis test: {str(e)}")
        traceback.print_exc()
        return None

def main():
    """Run all tests with different stocks"""
    print("\nTESTING ENHANCED PREDICTIVE ANALYTICS TOOLS")
    print("="*80)
    
    # Test with different stocks
    symbols = ['AAPL', 'MSFT', 'TSLA']
    
    for symbol in symbols:
        # Test anomaly detection
        anomaly_results = test_anomaly_detection(symbol, period='1y')
        
        # Test volatility analysis
        volatility_results = test_volatility_analysis(symbol, period='1y')
        
        # Test scenario analysis
        scenario_results = test_scenario_analysis(symbol, period='1y')
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()
