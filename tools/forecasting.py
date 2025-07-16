"""
Forecasting Tools for the Finance Analyst AI Agent.
Implements time series forecasting using statistical methods and machine learning.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Any, Optional, Tuple, Union
from langchain.tools import BaseTool
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

# Try importing optional dependencies
try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class StockPredictionTool(BaseTool):
    name = "stock_price_forecaster"
    description = """
    Forecasts future stock prices using time series models like ARIMA, Prophet, or LSTM.
    Provides predictions, confidence intervals, and trend analysis.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        model_type: Forecasting model to use ('arima', 'prophet', 'lstm', defaults to 'arima')
        days_ahead: Number of days to forecast (default: 30)
        history_period: Historical data period to use for training (e.g., '1y', '2y', defaults to '1y')
        
    Returns:
        A dictionary with forecast data, confidence intervals, and forecast analysis.
    """
    
    def _run(self, ticker: str, model_type: str = "arima", days_ahead: int = 30, history_period: str = "1y") -> Dict[str, Any]:
        try:
            # Fetch historical data
            stock_data = yf.Ticker(ticker)
            history = stock_data.history(period=history_period)
            
            if history.empty:
                return {"error": f"Could not fetch historical data for {ticker}"}
                
            # Extract closing prices
            closing_prices = history['Close']
            
            # Select forecasting method
            if model_type.lower() == "prophet":
                if not PROPHET_AVAILABLE:
                    return {"error": "Prophet is not installed. Please install with 'pip install prophet'."}
                forecast_result = self._forecast_with_prophet(closing_prices, days_ahead)
            elif model_type.lower() == "lstm":
                if not TF_AVAILABLE:
                    return {"error": "TensorFlow is not installed. Please install with 'pip install tensorflow'."}
                forecast_result = self._forecast_with_lstm(closing_prices, days_ahead)
            else:  # default to ARIMA
                if not STATSMODELS_AVAILABLE:
                    return {"error": "Statsmodels is not installed. Please install with 'pip install statsmodels'."}
                forecast_result = self._forecast_with_arima(closing_prices, days_ahead)
            
            # Add metadata
            forecast_result["ticker"] = ticker
            forecast_result["model_type"] = model_type
            forecast_result["forecast_date"] = datetime.now().strftime("%Y-%m-%d")
            forecast_result["history_period"] = history_period
            forecast_result["last_price"] = closing_prices.iloc[-1]
            forecast_result["historical_stats"] = {
                "mean": round(closing_prices.mean(), 2),
                "min": round(closing_prices.min(), 2),
                "max": round(closing_prices.max(), 2),
                "std_dev": round(closing_prices.std(), 2)
            }
            
            # Add insights based on the forecast
            forecast_result["insights"] = self._generate_forecast_insights(forecast_result, closing_prices.iloc[-1])
            
            return forecast_result
            
        except Exception as e:
            return {"error": f"Error forecasting stock price for {ticker}: {str(e)}"}
    
    def _forecast_with_arima(self, price_series: pd.Series, days_ahead: int) -> Dict[str, Any]:
        """Generate forecasts using ARIMA model."""
        # Fit ARIMA model - using auto_arima would be better but keeping it simple
        model = ARIMA(price_series, order=(5, 1, 0))
        model_fit = model.fit()
        
        # Generate forecast
        forecast = model_fit.forecast(steps=days_ahead)
        forecast_index = pd.date_range(start=price_series.index[-1] + pd.Timedelta(days=1), periods=days_ahead, freq='B')
        forecast = pd.Series(forecast, index=forecast_index)
        
        # Get confidence intervals
        pred_conf = model_fit.get_forecast(days_ahead).conf_int()
        lower_bound = pd.Series(pred_conf.iloc[:, 0].values, index=forecast_index)
        upper_bound = pd.Series(pred_conf.iloc[:, 1].values, index=forecast_index)
        
        # Prepare results
        forecast_dates = [date.strftime("%Y-%m-%d") for date in forecast.index]
        
        return {
            "forecast_dates": forecast_dates,
            "forecast_values": [round(x, 2) for x in forecast.values],
            "lower_bound": [round(x, 2) for x in lower_bound.values],
            "upper_bound": [round(x, 2) for x in upper_bound.values],
            "confidence_level": "95%"
        }
    
    def _forecast_with_prophet(self, price_series: pd.Series, days_ahead: int) -> Dict[str, Any]:
        """Generate forecasts using Facebook Prophet model."""
        # Prepare data for Prophet
        df = pd.DataFrame({'ds': price_series.index, 'y': price_series.values})
        
        # Create and fit model
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=days_ahead)
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Extract forecast dates (only for future predictions)
        future_forecast = forecast.iloc[-days_ahead:]
        forecast_dates = [date.strftime("%Y-%m-%d") for date in future_forecast['ds']]
        
        # Extract values and intervals
        forecast_values = [round(x, 2) for x in future_forecast['yhat'].values]
        lower_bound = [round(x, 2) for x in future_forecast['yhat_lower'].values]
        upper_bound = [round(x, 2) for x in future_forecast['yhat_upper'].values]
        
        return {
            "forecast_dates": forecast_dates,
            "forecast_values": forecast_values,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "confidence_level": "80%"  # Prophet uses 80% by default
        }
        
    def _forecast_with_lstm(self, price_series: pd.Series, days_ahead: int) -> Dict[str, Any]:
        """Generate forecasts using LSTM model."""
        # Normalize data
        data = price_series.values.reshape(-1, 1)
        data_mean = np.mean(data)
        data_std = np.std(data)
        normalized_data = (data - data_mean) / data_std
        
        # Prepare sequences for LSTM
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)
        
        # Use 60 days of history to predict next day
        seq_length = 60
        X, y = create_sequences(normalized_data, seq_length)
        
        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        
        # Generate future predictions
        last_sequence = normalized_data[-seq_length:].copy()
        forecast_values = []
        
        for _ in range(days_ahead):
            # Reshape for prediction
            current_sequence = last_sequence.reshape(1, seq_length, 1)
            
            # Predict next value
            next_pred = model.predict(current_sequence, verbose=0)[0][0]
            
            # Add to forecast
            forecast_values.append(next_pred)
            
            # Update sequence
            last_sequence = np.append(last_sequence[1:], next_pred).reshape(-1, 1)
        
        # Denormalize predictions
        forecast_values = [(x * data_std) + data_mean for x in forecast_values]
        
        # Generate dates
        last_date = price_series.index[-1]
        forecast_dates = [(last_date + pd.Timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(days_ahead)]
        
        # Create simple confidence intervals (fixed percentage from point estimate)
        forecast_values = [round(x, 2) for x in forecast_values]
        lower_bound = [round(x * 0.9, 2) for x in forecast_values]  # 10% below
        upper_bound = [round(x * 1.1, 2) for x in forecast_values]  # 10% above
        
        return {
            "forecast_dates": forecast_dates,
            "forecast_values": forecast_values,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "confidence_level": "approximate"  # Not a formal confidence interval
        }
    
    def _generate_forecast_insights(self, forecast_result: Dict[str, Any], last_price: float) -> List[str]:
        """Generate insights based on forecast results."""
        insights = []
        
        # Extract forecast info
        forecast_values = forecast_result.get("forecast_values", [])
        if not forecast_values:
            return ["Insufficient forecast data to generate insights."]
            
        # Calculate expected change
        first_forecast = forecast_values[0]
        last_forecast = forecast_values[-1]
        
        short_term_change = ((first_forecast / last_price) - 1) * 100
        overall_change = ((last_forecast / last_price) - 1) * 100
        
        # Short-term direction
        if short_term_change > 2:
            insights.append(f"The model forecasts a significant short-term increase of {round(short_term_change, 2)}% in the next few days.")
        elif short_term_change > 0.5:
            insights.append(f"The model forecasts a moderate short-term increase of {round(short_term_change, 2)}% in the next few days.")
        elif short_term_change < -2:
            insights.append(f"The model forecasts a significant short-term decrease of {round(abs(short_term_change), 2)}% in the next few days.")
        elif short_term_change < -0.5:
            insights.append(f"The model forecasts a moderate short-term decrease of {round(abs(short_term_change), 2)}% in the next few days.")
        else:
            insights.append("The model forecasts relatively stable prices in the short term.")
            
        # Overall direction
        if overall_change > 10:
            insights.append(f"Over the entire forecast period, the model projects a strong upward trend of {round(overall_change, 2)}%.")
        elif overall_change > 3:
            insights.append(f"Over the entire forecast period, the model projects a moderate upward trend of {round(overall_change, 2)}%.")
        elif overall_change < -10:
            insights.append(f"Over the entire forecast period, the model projects a strong downward trend of {round(abs(overall_change), 2)}%.")
        elif overall_change < -3:
            insights.append(f"Over the entire forecast period, the model projects a moderate downward trend of {round(abs(overall_change), 2)}%.")
        else:
            insights.append("Over the entire forecast period, the model projects relatively stable prices with minor fluctuations.")
            
        # Analyze volatility
        upper = forecast_result.get("upper_bound", [])
        lower = forecast_result.get("lower_bound", [])
        
        if upper and lower:
            # Calculate average spread as percentage
            spreads = [(u - l) / ((u + l) / 2) * 100 for u, l in zip(upper, lower)]
            avg_spread = sum(spreads) / len(spreads)
            
            if avg_spread > 15:
                insights.append(f"The forecast shows high uncertainty with an average spread of {round(avg_spread, 1)}% between upper and lower bounds.")
            elif avg_spread > 7:
                insights.append(f"The forecast shows moderate uncertainty with an average spread of {round(avg_spread, 1)}% between upper and lower bounds.")
            else:
                insights.append(f"The forecast shows relatively low uncertainty with an average spread of {round(avg_spread, 1)}% between upper and lower bounds.")
        
        # Add caution note
        model_type = forecast_result.get("model_type", "").upper()
        insights.append(f"Note: This {model_type} forecast is based on historical patterns and should be considered alongside fundamental factors, market news, and other analysis.")
        
        return insights
    
    async def _arun(self, ticker: str, model_type: str = "arima", days_ahead: int = 30, history_period: str = "1y") -> Dict[str, Any]:
        # Async implementation if needed
        return self._run(ticker, model_type, days_ahead, history_period)


class MacroEconomicForecastTool(BaseTool):
    name = "macro_economic_forecaster"
    description = """
    Forecasts economic indicators like GDP, inflation, or unemployment using time series models.
    Uses historical data from FRED to generate forecasts.
    
    Args:
        indicator: Economic indicator to forecast (e.g., 'GDP', 'INFLATION', 'UNEMPLOYMENT')
        quarters_ahead: Number of quarters to forecast (default: 4)
        model_type: Forecasting model to use ('arima', 'prophet', defaults to 'arima')
        
    Returns:
        A dictionary with forecast data, confidence intervals, and trend analysis.
    """
    
    # Mapping of common names to FRED series IDs (same as in economic_indicators.py)
    INDICATOR_MAPPING = {
        "GDP": "GDP",  # Gross Domestic Product
        "RGDP": "GDPC1",  # Real Gross Domestic Product
        "INFLATION": "CPIAUCSL",  # Consumer Price Index for All Urban Consumers
        "CORE_INFLATION": "CPILFESL",  # CPI Less Food and Energy
        "UNEMPLOYMENT": "UNRATE",  # Unemployment Rate
        "FED_FUNDS_RATE": "DFF",  # Federal Funds Effective Rate
        "YIELD_CURVE": "T10Y2Y",  # 10-Year Treasury Constant Maturity Minus 2-Year
        "RETAIL_SALES": "RSXFS",  # Retail Sales
        "INDUSTRIAL_PRODUCTION": "INDPRO",  # Industrial Production Index
        "HOUSING_STARTS": "HOUST",  # Housing Starts
        "CONSUMER_SENTIMENT": "UMCSENT",  # University of Michigan Consumer Sentiment
        "PCE": "PCE",  # Personal Consumption Expenditures
        "MONEY_SUPPLY": "M2SL",  # M2 Money Supply
    }
    
    def _run(self, indicator: str, quarters_ahead: int = 4, model_type: str = "arima") -> Dict[str, Any]:
        # Check for FRED API key
        from config import FRED_API_KEY
        if not FRED_API_KEY:
            return {"error": "FRED API key is not configured. Please set the FRED_API_KEY in your .env file."}
            
        # Check for required packages
        if not STATSMODELS_AVAILABLE:
            return {"error": "Statsmodels is not installed. Please install with 'pip install statsmodels'."}
            
        try:
            import fredapi
            from fredapi import Fred
            
            # Initialize FRED API
            fred = Fred(api_key=FRED_API_KEY)
            
            # Get series ID if common name is used
            series_id = self.INDICATOR_MAPPING.get(indicator.upper(), indicator)
            
            # Fetch historical data - get 10 years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * 10)  # 10 years of data
            
            series_data = fred.get_series(series_id, start_date, end_date)
            
            if series_data.empty:
                return {"error": f"No data found for indicator {indicator} (series ID: {series_id})"}
            
            # Get series information
            series_info = fred.get_series_info(series_id)
            
            # Select forecasting method
            if model_type.lower() == "prophet" and PROPHET_AVAILABLE:
                forecast_result = self._forecast_with_prophet(series_data, quarters_ahead)
            else:  # default to ARIMA
                forecast_result = self._forecast_with_arima(series_data, quarters_ahead)
                
            # Add metadata
            forecast_result["indicator"] = indicator
            forecast_result["indicator_name"] = series_info.title
            forecast_result["indicator_id"] = series_id
            forecast_result["model_type"] = model_type
            forecast_result["forecast_date"] = datetime.now().strftime("%Y-%m-%d")
            forecast_result["units"] = series_info.units
            forecast_result["frequency"] = series_info.frequency
            forecast_result["last_value"] = float(series_data.iloc[-1])
            forecast_result["last_date"] = series_data.index[-1].strftime("%Y-%m-%d")
            
            # Add insights
            forecast_result["insights"] = self._generate_forecast_insights(forecast_result, indicator)
            
            return forecast_result
            
        except Exception as e:
            return {"error": f"Error forecasting economic indicator {indicator}: {str(e)}"}
    
    def _forecast_with_arima(self, data_series: pd.Series, quarters_ahead: int) -> Dict[str, Any]:
        """Generate forecasts using ARIMA model."""
        # Convert quarters to time periods based on frequency
        freq = pd.infer_freq(data_series.index)
        if freq == 'Q':
            periods = quarters_ahead
        elif freq == 'M':
            periods = quarters_ahead * 3
        else:
            # Default to quarters_ahead if frequency can't be determined
            periods = quarters_ahead
            
        # Fit ARIMA model - simplified approach
        model = ARIMA(data_series, order=(2, 1, 2))
        model_fit = model.fit()
        
        # Generate forecast
        forecast = model_fit.forecast(steps=periods)
        
        # Create date range for forecast
        last_date = data_series.index[-1]
        if freq == 'Q':
            forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=3), periods=periods, freq='Q')
        elif freq == 'M':
            forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
        else:
            # Default to quarterly if frequency can't be determined
            forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=3), periods=periods, freq='Q')
            
        # Get confidence intervals
        pred_conf = model_fit.get_forecast(periods).conf_int()
        lower_bound = pd.Series(pred_conf.iloc[:, 0].values, index=forecast_index)
        upper_bound = pd.Series(pred_conf.iloc[:, 1].values, index=forecast_index)
        
        # Format dates
        forecast_dates = [date.strftime("%Y-%m-%d") for date in forecast_index]
        
        return {
            "forecast_dates": forecast_dates,
            "forecast_values": [round(float(x), 2) for x in forecast.values],
            "lower_bound": [round(float(x), 2) for x in lower_bound.values],
            "upper_bound": [round(float(x), 2) for x in upper_bound.values],
            "confidence_level": "95%"
        }
    
    def _forecast_with_prophet(self, data_series: pd.Series, quarters_ahead: int) -> Dict[str, Any]:
        """Generate forecasts using Facebook Prophet model."""
        # Convert quarters to time periods based on frequency
        freq = pd.infer_freq(data_series.index)
        if freq == 'Q':
            periods = quarters_ahead * 90  # approx days in quarter
        elif freq == 'M':
            periods = quarters_ahead * 90  # approx days in quarter
        else:
            # Default if frequency can't be determined
            periods = quarters_ahead * 90
            
        # Prepare data for Prophet
        df = pd.DataFrame({'ds': data_series.index, 'y': data_series.values})
        
        # Create and fit model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False)
        model.fit(df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods)
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Extract forecast for future dates only
        future_forecast = forecast[forecast['ds'] > data_series.index[-1]]
        
        # Resample to quarterly or monthly frequency
        if freq == 'Q':
            future_forecast = future_forecast.set_index('ds').resample('Q').mean().reset_index()
            future_forecast = future_forecast.head(quarters_ahead)
        elif freq == 'M':
            future_forecast = future_forecast.set_index('ds').resample('M').mean().reset_index()
            future_forecast = future_forecast.head(quarters_ahead * 3)
            
        # Extract dates and values
        forecast_dates = [date.strftime("%Y-%m-%d") for date in future_forecast['ds']]
        forecast_values = [round(float(x), 2) for x in future_forecast['yhat'].values]
        lower_bound = [round(float(x), 2) for x in future_forecast['yhat_lower'].values]
        upper_bound = [round(float(x), 2) for x in future_forecast['yhat_upper'].values]
        
        return {
            "forecast_dates": forecast_dates,
            "forecast_values": forecast_values,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "confidence_level": "80%"  # Prophet uses 80% by default
        }
    
    def _generate_forecast_insights(self, forecast_result: Dict[str, Any], indicator: str) -> List[str]:
        """Generate insights for economic forecasts."""
        insights = []
        
        # Get values
        indicator = indicator.upper()
        forecast_values = forecast_result.get("forecast_values", [])
        last_value = forecast_result.get("last_value")
        
        if not forecast_values or last_value is None:
            return ["Insufficient forecast data to generate insights."]
            
        # Calculate changes
        first_forecast = forecast_values[0]
        last_forecast = forecast_values[-1]
        
        short_term_change = ((first_forecast / last_value) - 1) * 100
        overall_trend = ((last_forecast / last_value) - 1) * 100
        
        # Generate insights based on indicator type
        if indicator in ["GDP", "RGDP"]:
            if overall_trend > 3:
                insights.append(f"The forecast projects strong GDP growth of {round(overall_trend, 2)}% over the forecast period.")
            elif overall_trend > 0:
                insights.append(f"The forecast projects modest GDP growth of {round(overall_trend, 2)}% over the forecast period.")
            elif overall_trend > -3:
                insights.append(f"The forecast projects a slight GDP contraction of {round(abs(overall_trend), 2)}% over the forecast period.")
            else:
                insights.append(f"The forecast projects a significant GDP contraction of {round(abs(overall_trend), 2)}% over the forecast period.")
                
            # Look for recession (two consecutive quarterly declines)
            if len(forecast_values) >= 2:
                consecutive_declines = sum(1 for i in range(1, len(forecast_values)) if forecast_values[i] < forecast_values[i-1])
                if consecutive_declines >= 2:
                    insights.append("The forecast shows patterns consistent with recessionary pressures.")
                    
        elif indicator in ["INFLATION", "CORE_INFLATION", "CPIAUCSL", "CPILFESL"]:
            if last_forecast > 4:
                insights.append(f"Inflation is forecasted to remain elevated at {round(last_forecast, 2)}% by the end of the period.")
            elif 2 <= last_forecast <= 4:
                insights.append(f"Inflation is forecasted to be moderate at {round(last_forecast, 2)}% by the end of the period.")
            else:
                insights.append(f"Inflation is forecasted to be relatively low at {round(last_forecast, 2)}% by the end of the period.")
                
            # Trend
            if overall_trend > 20:
                insights.append(f"The forecast projects a significant increase in inflation ({round(overall_trend, 2)}%), which may prompt monetary policy tightening.")
            elif overall_trend < -20:
                insights.append(f"The forecast projects a significant decrease in inflation ({round(abs(overall_trend), 2)}%), which may allow for monetary policy easing.")
                
        elif indicator in ["UNEMPLOYMENT", "UNRATE"]:
            if last_forecast < 4:
                insights.append(f"Unemployment is forecasted to be very low at {round(last_forecast, 2)}% by the end of the period, suggesting a tight labor market.")
            elif 4 <= last_forecast < 6:
                insights.append(f"Unemployment is forecasted to be moderate at {round(last_forecast, 2)}% by the end of the period.")
            else:
                insights.append(f"Unemployment is forecasted to be elevated at {round(last_forecast, 2)}% by the end of the period, suggesting labor market weakness.")
                
            # Trend
            if overall_trend > 15:
                insights.append(f"The forecast projects a significant increase in unemployment ({round(overall_trend, 2)}%), which may indicate economic contraction.")
            elif overall_trend < -15:
                insights.append(f"The forecast projects a significant decrease in unemployment ({round(abs(overall_trend), 2)}%), which suggests economic expansion.")
                
        elif indicator in ["FED_FUNDS_RATE", "DFF"]:
            if last_forecast > last_value and last_forecast > 3:
                insights.append(f"Interest rates are forecasted to rise to {round(last_forecast, 2)}%, suggesting a tightening monetary policy stance.")
            elif last_forecast < last_value and last_value > 2:
                insights.append(f"Interest rates are forecasted to decrease to {round(last_forecast, 2)}%, suggesting an easing monetary policy stance.")
            else:
                insights.append(f"Interest rates are forecasted to remain relatively stable around {round(last_forecast, 2)}%.")
                
        # Generic insights for other indicators
        else:
            if abs(overall_trend) < 5:
                insights.append(f"The forecast projects relatively stable trends in {indicator.lower()} over the forecast period.")
            elif overall_trend > 0:
                insights.append(f"The forecast projects an increase of {round(overall_trend, 2)}% in {indicator.lower()} over the forecast period.")
            else:
                insights.append(f"The forecast projects a decrease of {round(abs(overall_trend), 2)}% in {indicator.lower()} over the forecast period.")
                
        # Add model context
        model_type = forecast_result.get("model_type", "").upper()
        insights.append(f"Note: This {model_type} forecast is based on historical patterns and should be considered alongside other economic data and expert analyses.")
        
        return insights
    
    async def _arun(self, indicator: str, quarters_ahead: int = 4, model_type: str = "arima") -> Dict[str, Any]:
        # Async implementation if needed
        return self._run(indicator, quarters_ahead, model_type)
