"""
Alerts and Monitoring Tools for Finance Analyst AI Agent

This module provides real-time alerts, monitoring, and automation capabilities
for financial markets including:
- Price threshold alerts
- Technical indicator crossover alerts
- News sentiment alerts
- Economic calendar integration
- Automated reporting
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time
import threading
import json
import logging
from typing import Dict, List, Any, Optional, Union
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Import other tools
from tools.technical_analysis import TechnicalAnalysisTools
from tools.fundamental_analysis import FundamentalAnalysisTools

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlertsMonitoringTools:
    """Tools for financial alerts, monitoring, and automation"""
    
    # Class variables for alert storage
    active_alerts = {}
    alert_history = []
    monitoring_threads = {}
    
    @staticmethod
    def create_price_alert(symbol: str, target_price: float, condition: str, 
                          notification_method: str, expiration_days: int = 30) -> Dict:
        """
        Create a price alert for a stock
        
        Args:
            symbol: Stock symbol
            target_price: Target price to trigger alert
            condition: Condition for alert ('above' or 'below')
            notification_method: Method to send notification ('email', 'sms', 'slack')
            expiration_days: Number of days until alert expires
            
        Returns:
            Dictionary with alert details
        """
        try:
            # Validate inputs
            if condition not in ['above', 'below']:
                return {"error": "Condition must be 'above' or 'below'"}
            
            if notification_method not in ['email', 'sms', 'slack']:
                return {"error": "Notification method must be 'email', 'sms', or 'slack'"}
            
            # Get current price
            ticker = yf.Ticker(symbol)
            current_price = ticker.history(period="1d")['Close'].iloc[-1]
            
            # Generate alert ID
            alert_id = f"price_{symbol}_{condition}_{target_price}_{int(time.time())}"
            
            # Calculate expiration date
            expiration_date = datetime.now() + timedelta(days=expiration_days)
            
            # Create alert
            alert = {
                "id": alert_id,
                "type": "price",
                "symbol": symbol,
                "target_price": target_price,
                "current_price": current_price,
                "condition": condition,
                "notification_method": notification_method,
                "created_at": datetime.now().isoformat(),
                "expires_at": expiration_date.isoformat(),
                "status": "active",
                "triggered": False
            }
            
            # Store alert
            AlertsMonitoringTools.active_alerts[alert_id] = alert
            
            # Start monitoring thread if not already running
            if symbol not in AlertsMonitoringTools.monitoring_threads:
                thread = threading.Thread(
                    target=AlertsMonitoringTools._monitor_price,
                    args=(symbol,),
                    daemon=True
                )
                thread.start()
                AlertsMonitoringTools.monitoring_threads[symbol] = thread
            
            return {
                "success": True,
                "message": f"Price alert created for {symbol} {condition} ${target_price}",
                "alert_id": alert_id,
                "alert": alert
            }
            
        except Exception as e:
            logger.error(f"Error creating price alert: {str(e)}")
            return {"error": f"Error creating price alert: {str(e)}"}
    
    @staticmethod
    def create_indicator_alert(symbol: str, indicator: str, threshold: float, 
                              condition: str, notification_method: str, 
                              expiration_days: int = 30) -> Dict:
        """
        Create a technical indicator alert
        
        Args:
            symbol: Stock symbol
            indicator: Technical indicator ('rsi', 'macd', 'sma_cross', etc.)
            threshold: Threshold value to trigger alert
            condition: Condition for alert ('above', 'below', 'cross_above', 'cross_below')
            notification_method: Method to send notification ('email', 'sms', 'slack')
            expiration_days: Number of days until alert expires
            
        Returns:
            Dictionary with alert details
        """
        try:
            # Validate inputs
            valid_indicators = ['rsi', 'macd', 'sma_cross', 'ema_cross', 'bollinger']
            if indicator not in valid_indicators:
                return {"error": f"Indicator must be one of {valid_indicators}"}
            
            valid_conditions = ['above', 'below', 'cross_above', 'cross_below']
            if condition not in valid_conditions:
                return {"error": f"Condition must be one of {valid_conditions}"}
            
            if notification_method not in ['email', 'sms', 'slack']:
                return {"error": "Notification method must be 'email', 'sms', or 'slack'"}
            
            # Get current indicator value
            tech_data = TechnicalAnalysisTools.get_technical_signals(symbol)
            
            if "error" in tech_data:
                return tech_data
            
            # Get current indicator value based on indicator type
            current_value = None
            if indicator == 'rsi':
                current_value = tech_data['rsi']
            elif indicator == 'macd':
                current_value = tech_data['macd']
            elif indicator == 'sma_cross':
                current_value = tech_data['sma50'] / tech_data['sma200']
            elif indicator == 'bollinger':
                upper = tech_data['bollinger_upper']
                lower = tech_data['bollinger_lower']
                price = tech_data['current_price']
                # Calculate % position within bands
                current_value = (price - lower) / (upper - lower) * 100
            
            # Generate alert ID
            alert_id = f"indicator_{symbol}_{indicator}_{condition}_{threshold}_{int(time.time())}"
            
            # Calculate expiration date
            expiration_date = datetime.now() + timedelta(days=expiration_days)
            
            # Create alert
            alert = {
                "id": alert_id,
                "type": "indicator",
                "symbol": symbol,
                "indicator": indicator,
                "threshold": threshold,
                "current_value": current_value,
                "condition": condition,
                "notification_method": notification_method,
                "created_at": datetime.now().isoformat(),
                "expires_at": expiration_date.isoformat(),
                "status": "active",
                "triggered": False
            }
            
            # Store alert
            AlertsMonitoringTools.active_alerts[alert_id] = alert
            
            # Start monitoring thread if not already running
            monitor_key = f"{symbol}_indicator"
            if monitor_key not in AlertsMonitoringTools.monitoring_threads:
                thread = threading.Thread(
                    target=AlertsMonitoringTools._monitor_indicators,
                    args=(symbol,),
                    daemon=True
                )
                thread.start()
                AlertsMonitoringTools.monitoring_threads[monitor_key] = thread
            
            return {
                "success": True,
                "message": f"Indicator alert created for {symbol} {indicator} {condition} {threshold}",
                "alert_id": alert_id,
                "alert": alert
            }
            
        except Exception as e:
            logger.error(f"Error creating indicator alert: {str(e)}")
            return {"error": f"Error creating indicator alert: {str(e)}"}
