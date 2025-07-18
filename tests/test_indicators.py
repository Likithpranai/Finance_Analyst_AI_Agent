#!/usr/bin/env python3
"""
Test script for technical indicators
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging
from tools.technical_analysis import TechnicalAnalysisTools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_obv():
    """Test the On-Balance Volume indicator calculation"""
    try:
        # Get stock data
        symbol = "AAPL"
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="3mo")
        
        if data.empty:
            logger.error(f"Could not find data for {symbol}")
            return
        
        # Print data columns to verify
        logger.info(f"Data columns: {data.columns.tolist()}")
        logger.info(f"Data shape: {data.shape}")
        
        # Calculate OBV manually
        close_diff = data['Close'].diff()
        obv = pd.Series(0, index=data.index)
        obv.iloc[0] = data['Volume'].iloc[0]
        
        for i in range(1, len(data)):
            if close_diff.iloc[i] > 0:  # Price up, add volume
                obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i]
            elif close_diff.iloc[i] < 0:  # Price down, subtract volume
                obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i]
            else:  # Price unchanged, OBV unchanged
                obv.iloc[i] = obv.iloc[i-1]
        
        # Get the latest values and calculate change
        latest_obv = obv.iloc[-1]
        obv_change = ((obv.iloc[-1] - obv.iloc[-20]) / obv.iloc[-20]) * 100 if len(obv) >= 20 else 0
        
        logger.info(f"Manual OBV calculation successful")
        logger.info(f"Current OBV: {latest_obv:.0f}")
        logger.info(f"OBV 20-day Change: {obv_change:.2f}%")
        
        # Now try using the TechnicalAnalysisTools class
        try:
            tool_obv = TechnicalAnalysisTools.calculate_obv(data)
            logger.info(f"TechnicalAnalysisTools OBV calculation successful")
            logger.info(f"Current OBV (tool): {tool_obv.iloc[-1]:.0f}")
        except Exception as e:
            logger.error(f"Error using TechnicalAnalysisTools.calculate_obv: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in test_obv: {str(e)}")

def test_adline():
    """Test the Accumulation/Distribution Line indicator calculation"""
    try:
        # Get stock data
        symbol = "AAPL"
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="3mo")
        
        if data.empty:
            logger.error(f"Could not find data for {symbol}")
            return
        
        # Try using the TechnicalAnalysisTools class
        try:
            ad_line = TechnicalAnalysisTools.calculate_adline(data)
            logger.info(f"TechnicalAnalysisTools A/D Line calculation successful")
            logger.info(f"Current A/D Line: {ad_line.iloc[-1]:.2f}")
        except Exception as e:
            logger.error(f"Error using TechnicalAnalysisTools.calculate_adline: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in test_adline: {str(e)}")

def test_adx():
    """Test the Average Directional Index indicator calculation"""
    try:
        # Get stock data
        symbol = "AAPL"
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="3mo")
        
        if data.empty:
            logger.error(f"Could not find data for {symbol}")
            return
        
        # Try using the TechnicalAnalysisTools class
        try:
            adx_result = TechnicalAnalysisTools.calculate_adx(data)
            logger.info(f"TechnicalAnalysisTools ADX calculation successful")
            logger.info(f"Current ADX: {adx_result['ADX'].iloc[-1]:.2f}")
            logger.info(f"Current +DI: {adx_result['+DI'].iloc[-1]:.2f}")
            logger.info(f"Current -DI: {adx_result['-DI'].iloc[-1]:.2f}")
        except Exception as e:
            logger.error(f"Error using TechnicalAnalysisTools.calculate_adx: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in test_adx: {str(e)}")

if __name__ == "__main__":
    logger.info("Testing OBV calculation...")
    test_obv()
    
    logger.info("\nTesting A/D Line calculation...")
    test_adline()
    
    logger.info("\nTesting ADX calculation...")
    test_adx()
