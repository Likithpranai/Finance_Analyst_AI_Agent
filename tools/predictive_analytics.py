"""
Predictive Analytics Tools for Finance Analyst AI Agent

This module provides advanced predictive analytics capabilities including:
- Time series forecasting with Prophet
- LSTM-based predictions (when TensorFlow is available)
- Volatility modeling
- Anomaly detection
- Sentiment analysis integration
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.stattools import adfuller
import warnings

# Import Prophet for time series forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not available. Time series forecasting will be limited.")

# Try importing TensorFlow for LSTM models
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not available. LSTM predictions will not be available.")


class PredictiveAnalyticsTools:
    """Tools for advanced predictive analytics and forecasting"""
    
    @staticmethod
    def check_stationarity(time_series: pd.Series) -> Dict:
        """
        Check if a time series is stationary using the Augmented Dickey-Fuller test
        
        Args:
            time_series: Pandas Series with time series data
            
        Returns:
            Dictionary with test results
        """
        result = adfuller(time_series.dropna())
        
        return {
            "test_statistic": result[0],
            "p_value": result[1],
            "is_stationary": result[1] < 0.05,
            "critical_values": result[4],
            "interpretation": "Stationary" if result[1] < 0.05 else "Non-stationary"
        }
    
    @staticmethod
    def forecast_with_prophet(historical_data: pd.DataFrame, 
                             periods: int = 30, 
                             changepoint_prior_scale: float = 0.05,
                             yearly_seasonality: bool = True,
                             weekly_seasonality: bool = True,
                             daily_seasonality: bool = False) -> Dict:
        """
        Forecast future values using Facebook Prophet
        
        Args:
            historical_data: DataFrame with historical price data (yfinance format or other)
            periods: Number of periods to forecast
            changepoint_prior_scale: Controls flexibility of trend
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality
            daily_seasonality: Whether to include daily seasonality
            
        Returns:
            Dictionary with forecast results and model components
        """
        if not PROPHET_AVAILABLE:
            return {"error": "Prophet package not available"}
        
        try:
            # Check if we have valid data
            if historical_data is None or historical_data.empty:
                return {"error": "No historical data provided"}
                
            # Handle yfinance DataFrame format
            df = pd.DataFrame()
            
            # If index is DatetimeIndex, use it as ds
            if isinstance(historical_data.index, pd.DatetimeIndex):
                df['ds'] = historical_data.index
                if 'Close' in historical_data.columns:
                    df['y'] = historical_data['Close']
                elif len(historical_data.columns) > 0:
                    # Use the first numeric column if Close isn't available
                    for col in historical_data.columns:
                        if pd.api.types.is_numeric_dtype(historical_data[col]):
                            df['y'] = historical_data[col]
                            break
            # Handle standard format with explicit date column
            elif 'ds' in historical_data.columns and 'y' in historical_data.columns:
                df = historical_data.copy()
            elif 'Date' in historical_data.columns and 'Close' in historical_data.columns:
                df = historical_data.rename(columns={'Date': 'ds', 'Close': 'y'})
            else:
                return {"error": "Could not identify date and value columns in the data"}
            
            # Create and fit the model
            model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality
            )
            model.fit(df)
            
            # Make future dataframe and predict
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            # Calculate trend metrics
            trend_change = (forecast['trend'].iloc[-1] / forecast['trend'].iloc[0] - 1) * 100
            trend_direction = "Upward" if trend_change > 0 else "Downward"
            
            # Plot the forecast
            fig1 = model.plot(forecast)
            fig2 = model.plot_components(forecast)
            
            # Save plots
            forecast_plot_path = '/tmp/prophet_forecast.png'
            components_plot_path = '/tmp/prophet_components.png'
            fig1.savefig(forecast_plot_path)
            fig2.savefig(components_plot_path)
            plt.close(fig1)
            plt.close(fig2)
            
            # Extract relevant data for return
            forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_dict('records')
            
            return {
                "forecast_data": forecast_data,
                "trend_change_percent": float(trend_change),
                "trend_direction": trend_direction,
                "forecast_plot_path": forecast_plot_path,
                "components_plot_path": components_plot_path
            }
            
        except Exception as e:
            return {"error": f"Error in Prophet forecasting: {str(e)}"}
    
    @staticmethod
    def forecast_with_lstm(historical_data: pd.DataFrame, 
                          price_column: str = 'Close',
                          sequence_length: int = 60,
                          forecast_periods: int = 30,
                          train_split: float = 0.8,
                          epochs: int = 50,
                          batch_size: int = 32) -> Dict:
        """
        Forecast future values using LSTM neural network
        
        Args:
            historical_data: DataFrame with price data
            price_column: Column with price data
            sequence_length: Number of previous time steps to use as input features
            forecast_periods: Number of periods to forecast
            train_split: Proportion of data to use for training
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary with forecast results
        """
        if not TENSORFLOW_AVAILABLE:
            return {"error": "TensorFlow not available."}
        
        try:
            # Check if we have valid data
            if price_column not in historical_data.columns:
                return {"error": f"Price column '{price_column}' not found in data"}
            
            # Extract price data
            data = historical_data[price_column].values.reshape(-1, 1)
            
            # Normalize data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            
            # Create sequences for training
            X, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i, 0])
                y.append(scaled_data[i, 0])
            X, y = np.array(X), np.array(y)
            
            # Reshape X to be [samples, time steps, features]
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # Split into train and test sets
            train_size = int(len(X) * train_split)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Build LSTM model
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            
            # Compile and train
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            
            # Evaluate on test set
            y_pred = model.predict(X_test)
            rmse = np.sqrt(np.mean(np.square(y_pred - y_test)))
            
            # Forecast future values
            input_data = scaled_data[-sequence_length:].copy()
            future_predictions = []
            
            for i in range(forecast_periods):
                # Reshape for prediction
                x_input = input_data[-sequence_length:].reshape(1, sequence_length, 1)
                
                # Predict next value
                next_val = model.predict(x_input)[0][0]
                future_predictions.append(next_val)
                
                # Update input data for next prediction
                input_data = np.append(input_data, [[next_val]], axis=0)
            
            # Inverse transform to get actual values
            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            
            # Generate dates for forecast
            last_date = historical_data.index[-1] if isinstance(historical_data.index, pd.DatetimeIndex) else datetime.now()
            forecast_dates = pd.date_range(start=last_date, periods=forecast_periods+1)[1:]
            
            # Plot results
            plt.figure(figsize=(12, 6))
            
            # Plot historical data (last 100 points)
            hist_length = min(100, len(data))
            if isinstance(historical_data.index, pd.DatetimeIndex):
                plt.plot(historical_data.index[-hist_length:], data[-hist_length:], label='Historical')
            else:
                plt.plot(range(hist_length), data[-hist_length:], label='Historical')
            
            # Plot forecast
            plt.plot(forecast_dates, future_predictions, label='LSTM Forecast', color='red')
            plt.title('LSTM Forecast')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            
            # Save plot
            forecast_plot_path = '/tmp/lstm_forecast.png'
            plt.savefig(forecast_plot_path)
            plt.close()
            
            # Prepare forecast data
            forecast_data = [
                {"date": date.strftime('%Y-%m-%d'), "value": float(value[0])}
                for date, value in zip(forecast_dates, future_predictions)
            ]
            
            return {
                "forecast_data": forecast_data,
                "rmse": float(rmse),
                "forecast_plot_path": forecast_plot_path
            }
            
        except Exception as e:
            return {"error": f"Error in LSTM forecasting: {str(e)}"}
    
    @staticmethod
    def detect_anomalies(time_series: pd.Series, contamination: float = 0.05) -> Dict:
        """
        Detect anomalies in time series data using Isolation Forest
        
        Args:
            time_series: Pandas Series with time series data
            contamination: Expected proportion of anomalies
            
        Returns:
            Dictionary with anomaly detection results
        """
        try:
            # Prepare data
            data = time_series.values.reshape(-1, 1)
            
            # Create and fit the model
            model = IsolationForest(contamination=contamination, random_state=42)
            anomaly_labels = model.fit_predict(data)
            
            # Convert to binary labels (1: normal, -1: anomaly)
            anomalies = pd.Series(anomaly_labels == -1, index=time_series.index)
            
            # Get anomaly points
            anomaly_points = time_series[anomalies]
            
            # Calculate statistics
            anomaly_percent = anomalies.mean() * 100
            
            # Plot results
            plt.figure(figsize=(12, 6))
            plt.plot(time_series, label='Original Data')
            plt.scatter(anomaly_points.index, anomaly_points, color='red', label='Anomalies')
            plt.title('Anomaly Detection')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            
            # Save plot
            anomaly_plot_path = '/tmp/anomaly_detection.png'
            plt.savefig(anomaly_plot_path)
            plt.close()
            
            return {
                "anomaly_count": int(anomalies.sum()),
                "anomaly_percent": float(anomaly_percent),
                "anomaly_indices": anomalies[anomalies].index.tolist(),
                "anomaly_values": anomaly_points.tolist(),
                "anomaly_plot_path": anomaly_plot_path
            }
            
        except Exception as e:
            return {"error": f"Error in anomaly detection: {str(e)}"}
    
    @staticmethod
    def calculate_volatility(historical_data: pd.DataFrame, 
                            price_column: str = 'Close',
                            window_size: int = 20) -> Dict:
        """
        Calculate historical volatility and forecast future volatility
        
        Args:
            historical_data: DataFrame with price data
            price_column: Column with price data
            window_size: Window size for rolling volatility calculation
            
        Returns:
            Dictionary with volatility analysis results
        """
        try:
            # Calculate daily returns
            if price_column not in historical_data.columns:
                return {"error": f"Price column '{price_column}' not found in data"}
            
            returns = historical_data[price_column].pct_change().dropna()
            
            # Calculate rolling volatility (standard deviation of returns)
            rolling_vol = returns.rolling(window=window_size).std() * np.sqrt(252)  # Annualized
            
            # Calculate historical volatility statistics
            current_volatility = rolling_vol.iloc[-1]
            avg_volatility = rolling_vol.mean()
            max_volatility = rolling_vol.max()
            min_volatility = rolling_vol.min()
            
            # Determine volatility regime
            if current_volatility < avg_volatility * 0.8:
                volatility_regime = "Low"
            elif current_volatility > avg_volatility * 1.2:
                volatility_regime = "High"
            else:
                volatility_regime = "Normal"
            
            # Calculate volatility trend
            recent_trend = rolling_vol.iloc[-10:].mean() - rolling_vol.iloc[-30:-10].mean()
            volatility_trend = "Increasing" if recent_trend > 0 else "Decreasing"
            
            # Plot volatility
            plt.figure(figsize=(12, 6))
            plt.plot(rolling_vol.index, rolling_vol, label='Rolling Volatility')
            plt.axhline(y=avg_volatility, color='r', linestyle='-', label='Average Volatility')
            plt.title('Historical Volatility (Annualized)')
            plt.xlabel('Date')
            plt.ylabel('Volatility')
            plt.legend()
            
            # Save plot
            volatility_plot_path = '/tmp/volatility_analysis.png'
            plt.savefig(volatility_plot_path)
            plt.close()
            
            return {
                "current_volatility": float(current_volatility),
                "average_volatility": float(avg_volatility),
                "max_volatility": float(max_volatility),
                "min_volatility": float(min_volatility),
                "volatility_regime": volatility_regime,
                "volatility_trend": volatility_trend,
                "volatility_change_percent": float(((current_volatility / avg_volatility) - 1) * 100),
                "volatility_plot_path": volatility_plot_path
            }
            
        except Exception as e:
            return {"error": f"Error in volatility calculation: {str(e)}"}
    
    @staticmethod
    def scenario_analysis(historical_data: pd.DataFrame, 
                         price_column: str = 'Close',
                         scenarios: List[str] = ['base', 'bull', 'bear'],
                         forecast_periods: int = 30) -> Dict:
        """
        Perform scenario analysis for different market conditions
        
        Args:
            historical_data: DataFrame with price data
            price_column: Column with price data
            scenarios: List of scenarios to analyze
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with scenario analysis results
        """
        try:
            # Check if we have valid data
            if historical_data is None or historical_data.empty:
                return {"error": "No historical data provided"}
                
            # Calculate historical returns and volatility
            if price_column not in historical_data.columns:
                return {"error": f"Price column '{price_column}' not found in data"}
            
            returns = historical_data[price_column].pct_change().dropna()
            volatility = returns.std()
            last_price = historical_data[price_column].iloc[-1]
            
            # Define scenario parameters
            scenario_params = {
                'base': {'daily_return': returns.mean(), 'vol_multiplier': 1.0},
                'bull': {'daily_return': returns.mean() * 2, 'vol_multiplier': 0.8},
                'bear': {'daily_return': returns.mean() * -1, 'vol_multiplier': 1.5}
            }
            
            # Generate scenarios
            scenario_results = {}
            
            for scenario in scenarios:
                if scenario not in scenario_params:
                    continue
                
                # Get scenario parameters
                params = scenario_params[scenario]
                daily_return = params['daily_return']
                vol = volatility * params['vol_multiplier']
                
                # Simulate future prices using Monte Carlo
                simulations = 100  # Reduced from 1000 for performance
                simulation_df = pd.DataFrame()
                
                for i in range(simulations):
                    prices = [last_price]
                    
                    for j in range(forecast_periods):
                        # Generate random return from normal distribution
                        random_return = np.random.normal(daily_return, vol)
                        next_price = prices[-1] * (1 + random_return)
                        prices.append(next_price)
                    
                    simulation_df[i] = prices
                
                # Calculate statistics from simulations
                mean_path = simulation_df.mean(axis=1)
                upper_path = simulation_df.quantile(0.95, axis=1)
                lower_path = simulation_df.quantile(0.05, axis=1)
                
                # Generate dates for forecast
                try:
                    if isinstance(historical_data.index, pd.DatetimeIndex):
                        last_date = historical_data.index[-1]
                    elif 'Date' in historical_data.columns:
                        last_date = historical_data['Date'].iloc[-1]
                    else:
                        last_date = datetime.now()
                        
                    # Ensure last_date is a datetime object
                    if not isinstance(last_date, (datetime, pd.Timestamp)):
                        last_date = pd.to_datetime(last_date)
                        
                    forecast_dates = pd.date_range(start=last_date, periods=forecast_periods + 1)
                    
                    # Store results
                    scenario_results[scenario] = {
                        'dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
                        'mean_path': mean_path.tolist(),
                        'upper_path': upper_path.tolist(),
                        'lower_path': lower_path.tolist(),
                        'final_mean_price': float(mean_path.iloc[-1]),
                        'final_upper_price': float(upper_path.iloc[-1]),
                        'final_lower_price': float(lower_path.iloc[-1]),
                        'expected_return': float(((mean_path.iloc[-1] / last_price) - 1) * 100),
                        'worst_case_return': float(((lower_path.iloc[-1] / last_price) - 1) * 100),
                        'best_case_return': float(((upper_path.iloc[-1] / last_price) - 1) * 100)
                    }
                except Exception as e:
                    return {"error": f"Error in scenario analysis date handling: {str(e)}"}
            
            # Plot scenarios
            try:
                plt.figure(figsize=(12, 6))
                
                # Plot historical data - use last 30 days or all if less
                history_length = min(30, len(historical_data))
                
                if isinstance(historical_data.index, pd.DatetimeIndex):
                    historical_dates = historical_data.index[-history_length:]
                    historical_prices = historical_data[price_column].iloc[-history_length:]
                    plt.plot(historical_dates, historical_prices, label='Historical', color='black')
                
                # Plot each scenario
                colors = {'base': 'blue', 'bull': 'green', 'bear': 'red'}
                
                for scenario, results in scenario_results.items():
                    dates = pd.to_datetime(results['dates'])
                    plt.plot(dates, results['mean_path'], label=f'{scenario.capitalize()} Case', color=colors.get(scenario, 'gray'))
                    plt.fill_between(dates, results['lower_path'], results['upper_path'], alpha=0.2, color=colors.get(scenario, 'gray'))
                
                plt.title('Scenario Analysis')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                
                # Save plot
                scenario_plot_path = '/tmp/scenario_analysis.png'
                plt.savefig(scenario_plot_path)
                plt.close()
            except Exception as e:
                return {"error": f"Error in scenario analysis plotting: {str(e)}"}
            
            return {
                "scenarios": scenario_results,
                "scenario_plot_path": scenario_plot_path,
                "last_price": float(last_price),
                "forecast_periods": forecast_periods
            }
            
        except Exception as e:
            return {"error": f"Error in scenario analysis: {str(e)}"}
