import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler  # Use MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

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
    
    # Use MinMaxScaler instead of StandardScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Store feature names in the scaler for later use
    scaler.feature_names = X.columns.tolist()
    
    # Store results dictionary
    all_results = {}
    
    # Train Random Forest model
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    
    # Calculate RF metrics
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_rmse = np.sqrt(rf_mse)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_mape = np.mean(np.abs((y_test - rf_pred) / y_test)) * 100
    
    # Store RF results
    all_results['random_forest'] = {
        'model': rf_model,
        'mse': rf_mse,
        'rmse': rf_rmse,
        'mae': rf_mae,
        'r2': rf_r2,
        'mape': rf_mape,
        'pred': rf_pred
    }
    
    # Train XGBoost model
    print("Training XGBoost model...")
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    
    # Calculate XGB metrics
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    xgb_rmse = np.sqrt(xgb_mse)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_r2 = r2_score(y_test, xgb_pred)
    xgb_mape = np.mean(np.abs((y_test - xgb_pred) / y_test)) * 100
    
    # Store XGB results
    all_results['xgboost'] = {
        'model': xgb_model,
        'mse': xgb_mse,
        'rmse': xgb_rmse,
        'mae': xgb_mae,
        'r2': xgb_r2,
        'mape': xgb_mape,
        'pred': xgb_pred
    }
    
    # Add a simple moving average baseline
    ma_window = min(7, len(y_train))
    y_pred_ma = np.full_like(y_test, np.mean(y_train[-ma_window:]))
    
    # Calculate MA metrics
    ma_mse = mean_squared_error(y_test, y_pred_ma)
    ma_rmse = np.sqrt(ma_mse)
    ma_mae = mean_absolute_error(y_test, y_pred_ma)
    ma_r2 = r2_score(y_test, y_pred_ma)
    ma_mape = np.mean(np.abs((y_test - y_pred_ma) / y_test)) * 100
    
    # Store MA results
    all_results['moving_avg'] = {
        'model': None,  # No actual model for moving average
        'mse': ma_mse,
        'rmse': ma_rmse,
        'mae': ma_mae,
        'r2': ma_r2,
        'mape': ma_mape,
        'pred': y_pred_ma
    }
    
    # Select best model based on RMSE
    best_model_name = min(all_results.keys(), key=lambda x: all_results[x]['rmse'])
    print(f"Best model: {best_model_name}")
    
    # Create final results for best model
    best_results = all_results[best_model_name]
    
    # Create a DataFrame with actual vs predicted values
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': best_results['pred'],
        'error': y_test - best_results['pred'],
        'error_pct': ((y_test - best_results['pred']) / y_test) * 100
    })
    
    return {
        'model': best_results['model'] if best_model_name != 'moving_avg' else rf_model,  # Default to RF if MA is best
        'scaler': scaler,
        'features': X.columns.tolist(),
        'metrics': {
            'mse': best_results['mse'],
            'rmse': best_results['rmse'],
            'mae': best_results['mae'],
            'r2': best_results['r2'],
            'mape': best_results['mape']
        },
        'results': results_df,
        'best_model': best_model_name,
        'all_metrics': all_results
    }

def make_future_predictions(model, scaler, features, latest_data, days=5):
    """
    Make predictions for future days
    
    Args:
        model: Trained model
        scaler: Fitted scaler (MinMaxScaler expected)
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
        
        # Scale the features - create a DataFrame with proper column names
        if hasattr(scaler, 'feature_names'):
            # Use the saved feature names if available
            feature_df = pd.DataFrame(
                current_row[features].values,
                columns=scaler.feature_names
            )
            scaled_features = scaler.transform(feature_df)
        else:
            # Fallback to standard approach
            current_features = current_row[features].values
            scaled_features = scaler.transform(current_features)
        
        # Make prediction
        pred = model.predict(scaled_features)[0]
        predictions.append(pred)
        
        # Create a new row for the next day
        new_row = current_row.copy()
        
        # Update price-related features with prediction - Remove random noise
        new_row['close'] = pred
        # Estimate open/high/low based on predicted close and previous day's range
        prev_close = future_data.iloc[-1]['close']
        
        # Simple estimation: open near previous close, high/low based on predicted change
        # Ensure we use scalar values for comparison
        new_row['open'] = prev_close * 1.001 if pred > prev_close else prev_close * 0.999
        open_val = new_row['open'].iloc[0] if isinstance(new_row['open'], pd.Series) else new_row['open']
        
        new_row['high'] = max(pred, open_val) * 1.01 # Estimate high slightly above open/close
        new_row['low'] = min(pred, open_val) * 0.99   # Estimate low slightly below open/close
        
        # Ensure high >= open/close and low <= open/close
        # Extract scalar values before comparison
        high_val = new_row['high'].iloc[0] if isinstance(new_row['high'], pd.Series) else new_row['high']
        low_val = new_row['low'].iloc[0] if isinstance(new_row['low'], pd.Series) else new_row['low']
        close_val = new_row['close'].iloc[0] if isinstance(new_row['close'], pd.Series) else new_row['close']
        
        new_row['high'] = max(high_val, close_val, open_val)
        new_row['low'] = min(low_val, close_val, open_val)

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
            
        if 'ema5' in features:
            alpha = 2 / (5 + 1)
            new_row['ema5'] = (pred * alpha) + (future_data.iloc[-1]['ema5'] * (1 - alpha))
            
        if 'ema14' in features:
            alpha = 2 / (14 + 1)
            new_row['ema14'] = (pred * alpha) + (future_data.iloc[-1]['ema14'] * (1 - alpha))
        
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
            
        if 'volatility_ratio' in features and len(future_data) >= 4:
            recent_closes = list(future_data.tail(4)['close'].values) + [pred]
            volatility = np.std(recent_closes)
            new_row['volatility_ratio'] = volatility / pred if pred != 0 else 0 # Avoid division by zero
            
        # Update MA crossovers
        if 'ma_crossover' in features and 'ma5' in new_row and 'ma14' in new_row:
            ma5_val = new_row['ma5'].iloc[0] if isinstance(new_row['ma5'], pd.Series) else new_row['ma5']
            ma14_val = new_row['ma14'].iloc[0] if isinstance(new_row['ma14'], pd.Series) else new_row['ma14']
            new_row['ma_crossover'] = int(ma5_val > ma14_val)
            
        # Update Bollinger Bands
        if 'bb_upper' in features and 'ma14' in new_row:
            std_20 = np.std(list(future_data.tail(19)['close'].values) + [pred])
            ma14_val = new_row['ma14'].iloc[0] if isinstance(new_row['ma14'], pd.Series) else new_row['ma14']
            new_row['bb_upper'] = ma14_val + (std_20 * 2)
            
        if 'bb_lower' in features and 'ma14' in new_row:
            std_20 = np.std(list(future_data.tail(19)['close'].values) + [pred])
            ma14_val = new_row['ma14'].iloc[0] if isinstance(new_row['ma14'], pd.Series) else new_row['ma14']
            new_row['bb_lower'] = ma14_val - (std_20 * 2)
            
        if 'bb_width' in features and 'bb_upper' in new_row and 'bb_lower' in new_row and 'ma14' in new_row:
            bb_upper_val = new_row['bb_upper'].iloc[0] if isinstance(new_row['bb_upper'], pd.Series) else new_row['bb_upper']
            bb_lower_val = new_row['bb_lower'].iloc[0] if isinstance(new_row['bb_lower'], pd.Series) else new_row['bb_lower']
            ma14_val = new_row['ma14'].iloc[0] if isinstance(new_row['ma14'], pd.Series) else new_row['ma14']
            new_row['bb_width'] = (bb_upper_val - bb_lower_val) / ma14_val if ma14_val != 0 else 0 # Avoid division by zero
            
        # Update RSI if present
        if 'rsi' in features and len(future_data) >= 14:
            # Calculate price changes
            deltas = list(future_data.tail(14)['close'].diff().dropna().values)
            deltas.append(pred - future_data.iloc[-1]['close'])
            
            # Separate gains and losses
            gains = [max(d, 0) for d in deltas]
            losses = [abs(min(d, 0)) for d in deltas]
            
            # Calculate average gain and loss
            avg_gain = sum(gains) / 14
            avg_loss = sum(losses) / 14
            
            # Calculate RS and RSI
            if avg_loss == 0:
                new_row['rsi'] = 100
            else:
                rs = avg_gain / avg_loss
                new_row['rsi'] = 100 - (100 / (1 + rs))
                
        # Update momentum if present
        if 'momentum' in features:
            new_row['momentum'] = pred - future_data.iloc[-5]['close'] if len(future_data) >= 5 else 0
        
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
        historical_data: DataFrame with historical data (needs at least 'close' column)
        predictions: DataFrame with predictions (needs 'predicted_price' column)
        save_path: Path to save the visualization
    """
    # Get the last 30 days of historical data for better visualization
    last_n_days = 30
    if len(historical_data) > last_n_days:
        historical_subset = historical_data.iloc[-last_n_days:]
    else:
        historical_subset = historical_data
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(historical_subset.index, historical_subset['close'], 
             label='Historical Price', color='blue', linewidth=2)
    
    # Plot predictions
    plt.plot(predictions.index, predictions['predicted_price'], 
             label='Predicted Price', color='red', linestyle='--', linewidth=2)
    
    # Add scatter points for better visibility
    plt.scatter(historical_subset.index, historical_subset['close'], color='blue', s=30)
    plt.scatter(predictions.index, predictions['predicted_price'], color='red', s=30)
    
    # Add a vertical line at the last historical data point
    last_hist_date = historical_data.index[-1]
    last_hist_price = historical_data['close'].iloc[-1]
    plt.axvline(x=last_hist_date, color='gray', linestyle=':', linewidth=1)
    
    # Calculate price change percentage from the last historical price to the last predicted price
    last_pred_price = predictions['predicted_price'].iloc[-1]
    change_pct = ((last_pred_price - last_hist_price) / last_hist_price) * 100
    direction = "increase" if change_pct >= 0 else "decrease"
    
    # Add labels and title using the new calculation
    plt.title(f'Price Prediction: {abs(change_pct):.2f}% {direction} from last known price over next {len(predictions)} days', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()
    
    print(f"Prediction visualization saved as {save_path}")

def backtest_model(model, scaler, features, data, lookback_days=30, prediction_days=5):
    """
    Backtest the model on historical data
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        features: List of feature names
        data: DataFrame with historical data
        lookback_days: Number of days to use for each prediction
        prediction_days: Number of days ahead to predict
        
    Returns:
        DataFrame with backtest results
    """
    print("Backtesting model...")
    
    # Ensure we have enough data
    if len(data) < lookback_days + prediction_days + 20:
        print("Not enough data for backtesting")
        return None
    
    # Add feature names to scaler if not present
    if not hasattr(scaler, 'feature_names'):
        scaler.feature_names = features
    
    # Determine backtest points (every 10 days)
    backtest_points = list(range(lookback_days, len(data) - prediction_days, 10))
    
    results = []
    
    # Fix: iterate directly through backtest_points instead of using range()
    for i in backtest_points:
        # Get historical data up to this point
        historical = data.iloc[:i].copy()
        
        # Get actual future data for comparison
        actual = data.iloc[i:i+prediction_days].copy()
        
        if len(actual) < prediction_days:
            continue
        
        # Make prediction
        future_pred = make_future_predictions(
            model, scaler, features, historical, days=prediction_days
        )
        
        # Compare prediction with actual data
        for j in range(min(len(future_pred), len(actual))):
            pred_date = future_pred.index[j]
            # Find the closest date in actual data
            closest_date = actual.index[min(j, len(actual)-1)]
            
            pred_price = future_pred['predicted_price'].iloc[j]
            actual_price = actual['close'].loc[closest_date]
            
            error = actual_price - pred_price
            error_pct = (error / actual_price) * 100
            
            results.append({
                'prediction_date': historical.index[-1],
                'target_date': pred_date,
                'predicted_price': pred_price,
                'actual_price': actual_price,
                'error': error,
                'error_pct': error_pct
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Check if we have results before calculating metrics
    if len(results_df) == 0:
        print("No valid backtest results generated")
        return {
            'results': pd.DataFrame(),
            'metrics': {
                'mse': float('nan'),
                'rmse': float('nan'),
                'mae': float('nan'),
                'mape': float('nan')
            }
        }
    
    # Calculate aggregate metrics
    mse = mean_squared_error(results_df['actual_price'], results_df['predicted_price'])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(results_df['actual_price'], results_df['predicted_price'])
    mape = results_df['error_pct'].abs().mean()
    
    print(f"Backtest Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {
        'results': results_df,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    }
