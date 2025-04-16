import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def create_target(features_df, target_column='close', forecast_days=5):
    """
    Create target variable for prediction by shifting the target column
    
    Args:
        features_df: DataFrame with features
        target_column: Column to predict
        forecast_days: Number of days ahead to predict
        
    Returns:
        DataFrame with features and target
    """
    # Make a copy of the dataframe to avoid modifying the original
    df = features_df.copy()
    
    # Create target column (future price)
    df['target'] = df[target_column].shift(-forecast_days)
    
    # Drop rows with NaN targets (the last N rows where we can't calculate targets)
    df = df.dropna()
    
    return df

def train_model(features_df, target_column='close', forecast_days=5, test_size=0.2):
    """
    Train a model to predict future prices
    
    Args:
        features_df: DataFrame with features
        target_column: Column to predict
        forecast_days: Number of days ahead to predict
        test_size: Proportion of data to use for testing
        
    Returns:
        Dictionary with model, scaler, evaluation metrics and predictions
    """
    # Create the target variable
    data = create_target(features_df, target_column, forecast_days)
    
    # Separate features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False  # No shuffle for time series
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Create a DataFrame with actual vs predicted values
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'error': y_test - y_pred,
        'error_pct': ((y_test - y_pred) / y_test) * 100
    })
    
    return {
        'model': model,
        'scaler': scaler,
        'features': X.columns.tolist(),
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        },
        'results': results_df
    }

def make_future_predictions(model, scaler, features, latest_data, days=5):
    """
    Make predictions for future days
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        features: List of feature names
        latest_data: DataFrame with the most recent data
        days: Number of days to predict
        
    Returns:
        DataFrame with date and predicted prices
    """
    # Get the most recent data point
    base_data = latest_data.copy()
    
    # Store predictions
    predictions = []
    dates = []
    last_date = latest_data.index[-1]
    
    # Create a copy of the data for sequential predictions
    future_data = latest_data.tail(30).copy()  # Use last 30 days for proper feature calculation
    
    # Make predictions for each future day
    for i in range(days):
        # Calculate the next date
        future_date = last_date + timedelta(days=i+1)
        dates.append(future_date)
        
        # Get the features we need for prediction (most recent data point)
        current_row = future_data.iloc[-1:].copy()
        
        # Scale the features
        current_features = current_row[features].values
        scaled_features = scaler.transform(current_features)
        
        # Make prediction
        pred = model.predict(scaled_features)[0]
        predictions.append(pred)
        
        # Create a new row for the next day
        new_row = current_row.copy()
        
        # Update price-related features with prediction
        new_row['close'] = pred
        new_row['open'] = pred * (1 + np.random.normal(0, 0.01))  # Add slight randomness
        new_row['high'] = pred * (1 + abs(np.random.normal(0, 0.015)))  # Slightly higher
        new_row['low'] = pred * (1 - abs(np.random.normal(0, 0.01)))  # Slightly lower
        
        # Update moving averages properly
        if 'ma5' in features:
            recent_closes = list(future_data.tail(4)['close'].values) + [pred]
            new_row['ma5'] = sum(recent_closes) / 5
        
        if 'ma7' in features:
            recent_closes = list(future_data.tail(6)['close'].values) + [pred]
            new_row['ma7'] = sum(recent_closes) / 7
            
        if 'ma14' in features:
            recent_closes = list(future_data.tail(13)['close'].values) + [pred]
            new_row['ma14'] = sum(recent_closes) / 14
        
        # Update price changes
        if 'price_change_1d' in features:
            new_row['price_change_1d'] = (pred / future_data.iloc[-1]['close']) - 1
            
        if 'price_change_3d' in features and len(future_data) >= 3:
            new_row['price_change_3d'] = (pred / future_data.iloc[-3]['close']) - 1
            
        if 'price_change_7d' in features and len(future_data) >= 7:
            new_row['price_change_7d'] = (pred / future_data.iloc[-7]['close']) - 1
        
        # Update volatility
        if 'volatility_5d' in features and len(future_data) >= 4:
            recent_closes = list(future_data.tail(4)['close'].values) + [pred]
            new_row['volatility_5d'] = np.std(recent_closes)
        
        # Set the index for the new row
        new_row.index = [future_date]
        
        # Add the new row to the future data
        future_data = pd.concat([future_data, new_row])
    
    # Create prediction DataFrame
    prediction_df = pd.DataFrame({
        'date': dates,
        'predicted_price': predictions
    })
    prediction_df.set_index('date', inplace=True)
    
    return prediction_df

def visualize_predictions(historical_data, predictions, save_path="price_prediction.png"):
    """
    Visualize historical data and predictions
    
    Args:
        historical_data: DataFrame with historical data
        predictions: DataFrame with predictions
        save_path: Path to save the visualization
    """
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(historical_data.index, historical_data['close'], 
             label='Historical Price', color='blue')
    
    # Plot predictions
    plt.plot(predictions.index, predictions['predicted_price'], 
             label='Predicted Price', color='red', linestyle='--')
    
    # Add a vertical line at the last historical data point
    plt.axvline(x=historical_data.index[-1], color='gray', linestyle=':')
    
    # Add labels and title
    plt.title('Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()
    
    print(f"Prediction visualization saved as {save_path}")
