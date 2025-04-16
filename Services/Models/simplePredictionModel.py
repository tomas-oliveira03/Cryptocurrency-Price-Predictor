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
    base_data = latest_data[features].iloc[-1:].copy()
    
    # Store predictions
    predictions = []
    dates = []
    last_date = latest_data.index[-1]
    
    # Prepare initial features
    current_data = base_data.copy()
    
    # Make predictions for each future day
    for i in range(days):
        # Scale the features
        scaled_features = scaler.transform(current_data)
        
        # Make prediction
        pred = model.predict(scaled_features)[0]
        
        # Calculate date
        future_date = last_date + timedelta(days=i+1)
        dates.append(future_date)
        predictions.append(pred)
        
        # Update features for the next prediction 
        # (This is simplified - in a real system you'd have a more complex way to update features)
        # For now, just update the price-related features
        current_data['close'] = pred
        current_data['open'] = pred * 0.99  # Simple estimated open price
        current_data['high'] = pred * 1.02  # Simple estimated high price
        current_data['low'] = pred * 0.98   # Simple estimated low price
        
        # Update moving averages and other features as needed
        # This would require more sophisticated logic in a real system
    
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
