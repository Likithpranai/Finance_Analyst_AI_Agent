#!/usr/bin/env python3
"""
Test script for the predictive analytics integration in the Finance Analyst AI Agent.
This script tests the various predictive analytics capabilities including forecasting,
anomaly detection, volatility analysis, and scenario analysis.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from finance_analyst_agent import FinanceAnalystReActAgent

def test_predictive_analytics():
    """Test the predictive analytics integration in the Finance Analyst AI Agent"""
    print("Testing Predictive Analytics Integration")
    print("=======================================")
    
    # Initialize the agent
    agent = FinanceAnalystReActAgent()
    
    # Test queries for different predictive analytics capabilities
    test_queries = [
        # Forecasting with Prophet
        "Forecast AAPL stock price for the next month",
        
        # Volatility analysis
        "Analyze the volatility of MSFT over the past year",
        
        # Anomaly detection
        "Detect any anomalies in TSLA stock prices",
        
        # Scenario analysis
        "Run a scenario analysis for AMZN showing bull and bear cases",
        
        # Stationarity check
        "Check if GOOGL stock price is stationary"
    ]
    
    # Optional: Test LSTM if TensorFlow is available
    try:
        import tensorflow as tf
        test_queries.append("Use LSTM to forecast NVDA stock for the next 2 weeks")
        print("TensorFlow is available, will test LSTM forecasting")
    except ImportError:
        print("TensorFlow not available, skipping LSTM forecasting test")
    
    # Run each test query
    for i, query in enumerate(test_queries):
        print(f"\nTest {i+1}: {query}")
        print("-" * 50)
        
        try:
            # Process the query
            response = agent.process_query(query)
            
            # Print a truncated response (first 500 chars)
            print(f"Response (truncated): {response[:500]}...")
            print(f"Full response length: {len(response)} characters")
            
            # Check if there were any errors in the response
            if "error" in response.lower() and "not an error" not in response.lower():
                print("WARNING: Error detected in response")
            else:
                print("SUCCESS: Query processed without errors")
                
        except Exception as e:
            print(f"ERROR: Exception occurred: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\nPredictive Analytics Testing Complete")
    print("===================================")

if __name__ == "__main__":
    test_predictive_analytics()
