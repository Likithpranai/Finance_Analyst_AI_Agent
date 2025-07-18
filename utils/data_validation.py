"""
Data validation utilities for the Finance Analyst AI Agent.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def validate_stock_data(data, required_columns=None):
    """
    Validate that the provided data is a valid DataFrame with the required columns.
    
    Args:
        data: The data to validate
        required_columns: List of required columns (default: ['Open', 'High', 'Low', 'Close', 'Volume'])
    
    Returns:
        (bool, str): Tuple of (is_valid, error_message)
    """
    if required_columns is None:
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Check if data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        return False, f"Expected DataFrame, got {type(data)}"
    
    # Check if DataFrame is empty
    if data.empty:
        return False, "DataFrame is empty"
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    # Check for NaN values in required columns
    nan_columns = [col for col in required_columns if data[col].isna().any()]
    if nan_columns:
        logger.warning(f"NaN values found in columns: {nan_columns}")
    
    return True, "Data is valid"

def clean_stock_data(data):
    """
    Clean stock data by handling missing values and outliers.
    
    Args:
        data: DataFrame containing stock data
    
    Returns:
        Cleaned DataFrame
    """
    if not isinstance(data, pd.DataFrame):
        logger.error(f"Expected DataFrame, got {type(data)}")
        return data
    
    # Make a copy to avoid modifying the original
    cleaned_data = data.copy()
    
    # Forward fill missing values (use previous day's value)
    cleaned_data = cleaned_data.fillna(method='ffill')
    
    # If still have NaN at the beginning, backward fill
    cleaned_data = cleaned_data.fillna(method='bfill')
    
    # Handle any remaining NaNs with column means
    cleaned_data = cleaned_data.fillna(cleaned_data.mean())
    
    # Log how many values were filled
    nan_count = data.isna().sum().sum()
    if nan_count > 0:
        logger.info(f"Filled {nan_count} missing values in stock data")
    
    return cleaned_data
