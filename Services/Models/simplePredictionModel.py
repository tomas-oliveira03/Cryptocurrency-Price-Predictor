import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
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
    Train multiple models (RF, XGB, MA) on a train/test split of the provided data.
    Used for initial comparison or training on a subset (like the training fold).

    Args:
        features_df: DataFrame with features and target column.
        target_column: Column to predict.
        forecast_days: Number of days ahead the target variable represents.
        test_size: Proportion of data to use for internal testing within this function.

    Returns:
        Dictionary with evaluation metrics, models, predictions on the internal test set.
    """
    # Create the target variable
    data = create_target(features_df, target_column, forecast_days)

    if data.empty or 'target' not in data.columns:
        raise ValueError("Data is empty or 'target' column missing after create_target.")

    # Separate features and target
    X = data.drop('target', axis=1)
    y = data['target']

    if len(X) < 2: # Need at least 2 samples for train/test split
         raise ValueError(f"Not enough data ({len(X)} samples) to perform train/test split.")

    # Adjust test_size if it leads to a split of less than 1 sample
    if int(len(X) * test_size) < 1:
        test_size = 1 / len(X) # Ensure at least one test sample if possible
    if int(len(X) * (1 - test_size)) < 1:
         raise ValueError("Test size too large, leaves no training samples.")


    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False  # No shuffle for time series
    )

    if X_train.empty or X_test.empty:
         raise ValueError("Train or test split resulted in empty DataFrame.")

    # Use MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Store feature names
    scaler.feature_names = X.columns.tolist()

    # Store results dictionary
    all_results = {}

    # --- Train Random Forest ---
    try:
        print("Training Random Forest model...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # Use more cores
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_metrics = calculate_metrics(y_test, rf_pred)
        all_results['random_forest'] = {'model': rf_model, 'pred': rf_pred, **rf_metrics}
    except Exception as e:
        print(f"Error training Random Forest: {e}")
        all_results['random_forest'] = {'model': None, 'pred': np.full(len(y_test), np.nan), 'rmse': np.inf}

    # --- Train XGBoost ---
    try:
        print("Training XGBoost model...")
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror', n_estimators=100, learning_rate=0.1,
            max_depth=5, random_state=42, n_jobs=-1 # Use more cores
        )
        xgb_model.fit(X_train_scaled, y_train)
        xgb_pred = xgb_model.predict(X_test_scaled)
        xgb_metrics = calculate_metrics(y_test, xgb_pred)
        all_results['xgboost'] = {'model': xgb_model, 'pred': xgb_pred, **xgb_metrics}
    except Exception as e:
        print(f"Error training XGBoost: {e}")
        all_results['xgboost'] = {'model': None, 'pred': np.full(len(y_test), np.nan), 'rmse': np.inf}


    # --- Moving Average Baseline ---
    ma_window = min(7, len(y_train))
    if ma_window > 0:
        # Predict next value based on rolling mean of training data
        # For multi-step forecast, a simple approach is to hold the last mean constant
        last_ma_value = y_train.rolling(window=ma_window).mean().iloc[-1]
        if pd.isna(last_ma_value): # Handle case where window is larger than train set
             last_ma_value = y_train.mean()
        y_pred_ma = np.full(len(y_test), last_ma_value)

        ma_metrics = calculate_metrics(y_test, y_pred_ma)
        all_results['moving_avg'] = {'model': None, 'window': ma_window, 'pred': y_pred_ma, **ma_metrics}
    else:
        print("Not enough training data for Moving Average.")
        all_results['moving_avg'] = {'model': None, 'window': 0, 'pred': np.full(len(y_test), np.nan), 'rmse': np.inf}


    # Select best model based on RMSE from this internal test run
    valid_models = {k: v for k, v in all_results.items() if v.get('rmse', np.inf) != np.inf}
    if not valid_models:
         raise ValueError("All simple models failed to train.")

    best_model_name = min(valid_models.keys(), key=lambda x: valid_models[x]['rmse'])
    print(f"Best model (internal test): {best_model_name}")

    best_results = valid_models[best_model_name]

    # Create a DataFrame with actual vs predicted values for the best model
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': best_results['pred'],
    })
    results_df['error'] = results_df['actual'] - results_df['predicted']
    results_df['error_pct'] = (results_df['error'] / results_df['actual'].replace(0, np.nan)) * 100 # Avoid div by zero


    return {
        # Return the best performing model object (or RF as default if MA is best)
        'model': best_results.get('model') if best_model_name != 'moving_avg' else all_results.get('random_forest', {}).get('model'),
        'scaler': scaler,
        'features': X.columns.tolist(),
        'metrics': {k: v for k, v in best_results.items() if k not in ['model', 'pred']},
        'results': results_df, # Results on the internal test set
        'best_model': best_model_name,
        'all_metrics': all_results, # Contains all models and their preds on internal test
        'X_test': X_test,
        'y_test': y_test
    }

def train_specific_model(model_name, features_df, target_column='close', window=7):
    """
    Train a specific model (RF, XGB, or MA) on the *entire* provided dataset.
    Used for retraining the selected best model on all historical data.

    Args:
        model_name (str): 'random_forest', 'xgboost', or 'moving_avg'.
        features_df (pd.DataFrame): The full feature dataset.
        target_column (str): The name of the target column (e.g., 'close').
        window (int): Window size, primarily for 'moving_avg'.

    Returns:
        dict: Contains 'model', 'scaler', 'features', and potentially 'window'.
              Returns None for 'model' if model_name is 'moving_avg'.
    """
    print(f"\nRetraining {model_name.upper()} on full dataset ({len(features_df)} points)...")

    # MA doesn't require training in the traditional sense, just the window param.
    if model_name == 'moving_avg':
        # We still need a scaler and features list for consistency in prediction function
        temp_df = features_df.drop(columns=[target_column], errors='ignore')
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit scaler on dummy data of the correct shape if df is empty, else fit on actual data
        if not temp_df.empty:
             scaler.fit(temp_df)
             features = temp_df.columns.tolist()
        else:
             # Create dummy data with expected feature count if possible
             # This part is tricky without knowing expected features beforehand.
             # Assuming features_df has columns even if empty rows.
             features = [col for col in features_df.columns if col != target_column]
             if features:
                  dummy_data = pd.DataFrame(np.zeros((1, len(features))), columns=features)
                  scaler.fit(dummy_data)
             else:
                  print("Warning: Cannot determine features for MA scaler fitting on empty data.")
                  features = [] # No features to scale

        scaler.feature_names = features
        print(f"Moving Average selected. Window size: {window}")
        return {'model': None, 'scaler': scaler, 'features': features, 'window': window}

    # For RF and XGB:
    # We don't need a target variable shifted here, as we train on historical features
    # to predict the 'close' price directly in make_future_predictions.
    X = features_df.drop(columns=[target_column], errors='ignore')
    y = features_df[target_column] # The actual historical prices

    if X.empty or y.empty:
        raise ValueError(f"Cannot retrain {model_name}: Input data (X or y) is empty.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    scaler.feature_names = X.columns.tolist()

    model = None
    if model_name == 'random_forest':
        try:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_scaled, y)
            print("Random Forest retrained successfully.")
        except Exception as e:
            print(f"Error retraining Random Forest: {e}")
            raise # Re-raise the exception

    elif model_name == 'xgboost':
        try:
            model = xgb.XGBRegressor(
                objective='reg:squarederror', n_estimators=100, learning_rate=0.1,
                max_depth=5, random_state=42, n_jobs=-1
            )
            model.fit(X_scaled, y)
            print("XGBoost retrained successfully.")
        except Exception as e:
            print(f"Error retraining XGBoost: {e}")
            raise # Re-raise the exception
    else:
        raise ValueError(f"Unknown model_name for retraining: {model_name}")

    return {'model': model, 'scaler': scaler, 'features': X.columns.tolist()}

def calculate_metrics(y_true, y_pred):
    """Calculate standard regression metrics."""
    y_true = pd.Series(y_true).values # Ensure numpy array
    y_pred = pd.Series(y_pred).values # Ensure numpy array

    # Remove NaNs resulting from alignment or failed predictions
    valid_idx = pd.notna(y_true) & pd.notna(y_pred)
    if not np.any(valid_idx):
        return {'mse': np.inf, 'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf, 'mape': np.inf}

    y_true_valid = y_true[valid_idx]
    y_pred_valid = y_pred[valid_idx]

    if len(y_true_valid) == 0:
         return {'mse': np.inf, 'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf, 'mape': np.inf}


    mse = mean_squared_error(y_true_valid, y_pred_valid)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_valid, y_pred_valid)
    r2 = r2_score(y_true_valid, y_pred_valid)

    # Calculate MAPE carefully
    mape_mask = y_true_valid != 0
    if np.any(mape_mask):
        mape = np.mean(np.abs((y_true_valid[mape_mask] - y_pred_valid[mape_mask]) / y_true_valid[mape_mask])) * 100
    else:
        mape = np.inf

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

def predict_moving_average(series, window, steps):
     """Predict future values using a simple moving average."""
     if window is None or window <= 0 or len(series) < window:
          # Fallback: use the last known value if MA cannot be calculated
          last_value = series.iloc[-1] if not series.empty else 0
          return np.full(steps, last_value)
     # Calculate the last moving average value
     last_ma = series.rolling(window=window).mean().iloc[-1]
     # Predict the same value for all future steps
     return np.full(steps, last_ma)

def make_future_predictions(model, scaler, features, latest_data, days=5, window=None):
    """
    Make predictions for future days using RF, XGB, or MA.

    Args:
        model: Trained model (RF or XGB), or None for MA.
        scaler: Fitted scaler (MinMaxScaler expected).
        features: List of feature names the model/scaler expects.
        latest_data: DataFrame with the most recent historical data (needs all feature columns).
        days: Number of days to predict.
        window: The moving average window size (only used if model is None).

    Returns:
        DataFrame with date and predicted prices.
    """
    if latest_data.empty:
        print("Warning: latest_data is empty in make_future_predictions. Returning empty DataFrame.")
        return pd.DataFrame(columns=['predicted_price'])

    # Ensure 'features' list only contains columns present in latest_data
    available_features = [f for f in features if f in latest_data.columns]
    if len(available_features) != len(features):
        missing = set(features) - set(available_features)
        print(f"Warning: Missing features in latest_data for prediction: {missing}. Using available features only.")
        # Update scaler's feature list if necessary (though ideally scaler is already fit correctly)
        if hasattr(scaler, 'feature_names') and set(scaler.feature_names) != set(available_features):
             print("Warning: Scaler feature names mismatch available features. Prediction might be inaccurate.")
             # Attempt to proceed with available features, assuming scaler can handle it or was fit on them.
             features = available_features # Use only available features

    if not features:
         print("Error: No features available for prediction.")
         return pd.DataFrame(columns=['predicted_price'])


    # Get the most recent data point(s) needed for feature calculation lookback
    # Use a larger buffer to handle lagged features correctly
    lookback_needed = 30 # Adjust if longer lags are used
    base_data = latest_data.tail(lookback_needed).copy()

    predictions = []
    dates = []
    last_date = base_data.index[-1]

    # --- Moving Average Prediction ---
    if model is None:
        print(f"Making future predictions using Moving Average (window={window})...")
        # Use the dedicated MA prediction function
        ma_predictions = predict_moving_average(latest_data['close'], window, days)
        for i in range(days):
            future_date = last_date + timedelta(days=i + 1)
            dates.append(future_date)
            predictions.append(ma_predictions[i])

        prediction_df = pd.DataFrame({'date': dates, 'predicted_price': predictions})
        prediction_df.set_index('date', inplace=True)
        return prediction_df

    # --- RF / XGB Prediction ---
    print(f"Making future predictions using {type(model).__name__}...")
    future_data = base_data.copy() # Dataframe to update iteratively

    for i in range(days):
        future_date = last_date + timedelta(days=i + 1)
        dates.append(future_date)

        # Get the features from the *last available row* in our iteratively updated future_data
        current_features_row = future_data.iloc[-1:]

        # Ensure the row contains all necessary features before scaling
        current_features_values = current_features_row[features]

        # Scale the features
        # Check if scaler expects DataFrame or NumPy array
        try:
             # Some scalers might expect DataFrame with named columns
             scaled_features = scaler.transform(current_features_values)
        except TypeError:
             # Others might expect NumPy array
             scaled_features = scaler.transform(current_features_values.values)
        except Exception as e:
             print(f"Error during scaling: {e}")
             # Handle error, maybe append NaN or last prediction
             predictions.append(predictions[-1] if predictions else np.nan)
             # Create placeholder row to continue loop if needed
             new_row = future_data.iloc[-1:].copy()
             new_row.index = [future_date]
             future_data = pd.concat([future_data, new_row])
             continue


        # Make prediction
        try:
            pred = model.predict(scaled_features)[0]
            predictions.append(pred)
        except Exception as e:
             print(f"Error during prediction: {e}")
             predictions.append(predictions[-1] if predictions else np.nan)
             # Create placeholder row to continue loop if needed
             new_row = future_data.iloc[-1:].copy()
             new_row.index = [future_date]
             future_data = pd.concat([future_data, new_row])
             continue


        # --- Update features for the next step's prediction ---
        # Create a new row representing the predicted state
        new_row = current_features_row.copy()
        new_row.index = [future_date] # Set index to the future date

        # Update the 'close' price with the prediction
        new_row['close'] = pred

        # Update other features based on this new 'close' price
        # This requires re-calculating features that depend on 'close' or time
        # Example: Update moving averages using the new predicted price

        # Important: Re-calculate features based on the *updated* future_data history
        temp_history = pd.concat([future_data, new_row]) # Include the new row temporarily

        # Recalculate MAs
        if 'ma5' in features:
            new_row['ma5'] = temp_history['close'].rolling(window=5).mean().iloc[-1]
        if 'ma7' in features:
            new_row['ma7'] = temp_history['close'].rolling(window=7).mean().iloc[-1]
        if 'ma14' in features:
            new_row['ma14'] = temp_history['close'].rolling(window=14).mean().iloc[-1]
        # Add other feature recalculations here (EMA, RSI, Volatility, etc.)
        # This part needs to mirror the logic in engineFeatures.py carefully!
        # --- Simplified example for price change ---
        if 'price_change_1d' in features and len(future_data) > 0:
             new_row['price_change_1d'] = (pred / future_data['close'].iloc[-1]) - 1 if future_data['close'].iloc[-1] != 0 else 0

        # Add more feature updates as needed...

        # Append the updated row for the next iteration's calculation
        future_data = pd.concat([future_data, new_row])


    prediction_df = pd.DataFrame({'date': dates, 'predicted_price': predictions})
    prediction_df.set_index('date', inplace=True)
    return prediction_df

def visualize_predictions(historical_data, predictions, sentiment_data=None, save_path="price_prediction.png"):
    """
    Visualize historical data, predictions, and optionally sentiment
    
    Args:
        historical_data: DataFrame with historical data (needs at least 'close' column)
        predictions: DataFrame with predictions (needs 'predicted_price' column)
        sentiment_data: Optional DataFrame with sentiment data (needs index matching historical_data and a 'pct_positive' column)
        save_path: Path to save the visualization
    """
    # Get the last 30 days of historical data for better visualization
    last_n_days = 30
    if len(historical_data) > last_n_days:
        historical_subset = historical_data.iloc[-last_n_days:]
    else:
        historical_subset = historical_data
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(14, 7)) # Increased figure size slightly
    
    # Plot historical data on primary axis
    ax1.plot(historical_subset.index, historical_subset['close'], 
             label='Historical Price', color='blue', linewidth=2)
    
    # Plot predictions on primary axis
    ax1.plot(predictions.index, predictions['predicted_price'], 
             label='Predicted Price', color='red', linestyle='--', linewidth=2)
    
    # Add scatter points for better visibility
    ax1.scatter(historical_subset.index, historical_subset['close'], color='blue', s=30)
    ax1.scatter(predictions.index, predictions['predicted_price'], color='red', s=30)
    
    # Add a vertical line at the last historical data point
    last_hist_date = historical_data.index[-1]
    last_hist_price = historical_data['close'].iloc[-1]
    ax1.axvline(x=last_hist_date, color='gray', linestyle=':', linewidth=1)
    
    # Set primary y-axis label
    ax1.set_ylabel('Price ($)', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3) # Grid only for price axis
    
    # Calculate price change percentage for title
    last_pred_price = predictions['predicted_price'].iloc[-1]
    change_pct = ((last_pred_price - last_hist_price) / last_hist_price) * 100
    direction = "increase" if change_pct >= 0 else "decrease"
    
    # Add sentiment data if provided
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = [], []
    if sentiment_data is not None and not sentiment_data.empty and 'pct_positive' in sentiment_data.columns:
        # Align sentiment data with the historical subset dates
        sentiment_subset = sentiment_data.reindex(historical_subset.index).dropna()
        
        if not sentiment_subset.empty:
            # Create secondary axis
            ax2 = ax1.twinx()
            
            # Plot positive sentiment percentage on secondary axis
            ax2.plot(sentiment_subset.index, sentiment_subset['pct_positive'], 
                     label='Positive Sentiment Ratio', color='green', linestyle=':', linewidth=1.5, alpha=0.7)
            
            # Set secondary y-axis label and limits
            ax2.set_ylabel('Positive Sentiment Ratio (0-1)', color='green', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='green')
            ax2.set_ylim(0, 1) # Sentiment ratio is between 0 and 1
            
            # Get legend items for the second axis
            lines2, labels2 = ax2.get_legend_handles_labels()

    # Combine legends
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    # Add title and common x-axis label
    plt.title(f'Price Prediction & Sentiment: {abs(change_pct):.2f}% {direction} from last known price over next {len(predictions)} days', 
              fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    
    # Format x-axis
    fig.autofmt_xdate()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()
    
    print(f"Prediction visualization saved as {save_path}")