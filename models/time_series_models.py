"""
Time series forecasting models for the Finance Analyst AI Agent.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import warnings

# Optional imports with fallbacks
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not available. Prophet models will not be available.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow not available. Deep learning models will not be available.")

logger = logging.getLogger(__name__)

class TimeSeriesForecaster:
    """Base class for time series forecasting models."""
    
    def __init__(self, model_type="prophet"):
        """
        Initialize the forecaster.
        
        Args:
            model_type: Type of model to use ("prophet", "lstm", "arima")
        """
        self.model_type = model_type
        self.model = None
        self.is_fitted = False
        
        # Check if required packages are available
        if model_type == "prophet" and not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required for prophet model but not available")
        if model_type == "lstm" and not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model but not available")
    
    def fit(self, data, target_column="Close", **kwargs):
        """
        Fit the forecasting model.
        
        Args:
            data: DataFrame with datetime index and target column
            target_column: Column to forecast
            **kwargs: Additional arguments for specific models
        
        Returns:
            self: The fitted model
        """
        if self.model_type == "prophet":
            return self._fit_prophet(data, target_column, **kwargs)
        elif self.model_type == "lstm":
            return self._fit_lstm(data, target_column, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def predict(self, periods=30, **kwargs):
        """
        Generate forecasts.
        
        Args:
            periods: Number of periods to forecast
            **kwargs: Additional arguments for specific models
        
        Returns:
            DataFrame with forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.model_type == "prophet":
            return self._predict_prophet(periods, **kwargs)
        elif self.model_type == "lstm":
            return self._predict_lstm(periods, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _fit_prophet(self, data, target_column="Close", **kwargs):
        """Fit Prophet model."""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required but not available")
        
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_data = data.reset_index().rename(columns={
            data.index.name if data.index.name else 'index': 'ds',
            target_column: 'y'
        })
        
        # Create and fit the model
        self.model = Prophet(
            daily_seasonality=kwargs.get('daily_seasonality', True),
            weekly_seasonality=kwargs.get('weekly_seasonality', True),
            yearly_seasonality=kwargs.get('yearly_seasonality', True),
            interval_width=kwargs.get('interval_width', 0.95)
        )
        
        # Add additional regressors if provided
        regressors = kwargs.get('regressors', [])
        for regressor in regressors:
            if regressor in prophet_data.columns:
                self.model.add_regressor(regressor)
        
        # Fit the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(prophet_data)
        
        self.is_fitted = True
        self.target_column = target_column
        self.last_date = prophet_data['ds'].max()
        
        return self
    
    def _predict_prophet(self, periods=30, **kwargs):
        """Generate forecasts using Prophet."""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required but not available")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq='D')
        
        # Add additional regressors for future periods if provided
        future_regressors = kwargs.get('future_regressors', {})
        for regressor, values in future_regressors.items():
            if regressor in self.model.extra_regressors:
                future[regressor] = values
        
        # Make predictions
        forecast = self.model.predict(future)
        
        # Format the output
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
            'ds': 'Date',
            'yhat': f'Forecast_{self.target_column}',
            'yhat_lower': f'Lower_Bound_{self.target_column}',
            'yhat_upper': f'Upper_Bound_{self.target_column}'
        })
        
        result = result.set_index('Date')
        
        # Only return future predictions if requested
        if kwargs.get('future_only', True):
            result = result[result.index > self.last_date]
        
        return result
    
    def _fit_lstm(self, data, target_column="Close", **kwargs):
        """Fit LSTM model."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required but not available")
        
        # Extract parameters
        sequence_length = kwargs.get('sequence_length', 10)
        epochs = kwargs.get('epochs', 50)
        batch_size = kwargs.get('batch_size', 32)
        
        # Prepare data
        values = data[target_column].values.reshape(-1, 1)
        
        # Normalize data
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = self.scaler.fit_transform(values)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_values) - sequence_length):
            X.append(scaled_values[i:i+sequence_length])
            y.append(scaled_values[i+sequence_length])
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build model
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0
        )
        
        self.is_fitted = True
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.last_sequence = scaled_values[-sequence_length:]
        self.last_date = data.index[-1]
        
        return self
    
    def _predict_lstm(self, periods=30, **kwargs):
        """Generate forecasts using LSTM."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required but not available")
        
        # Initialize with last known sequence
        current_sequence = self.last_sequence.copy()
        predictions = []
        
        # Generate predictions one by one
        for _ in range(periods):
            # Reshape for prediction
            current_batch = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Predict next value
            next_pred = self.model.predict(current_batch, verbose=0)
            
            # Add to predictions
            predictions.append(next_pred[0, 0])
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], next_pred)[..., np.newaxis]
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        # Create date range for predictions
        last_date = self.last_date
        dates = [last_date + timedelta(days=i+1) for i in range(periods)]
        
        # Create result DataFrame
        result = pd.DataFrame({
            'Date': dates,
            f'Forecast_{self.target_column}': predictions.flatten()
        })
        
        result = result.set_index('Date')
        
        return result
