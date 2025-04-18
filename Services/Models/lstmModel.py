import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def create_sequences(X, y, seq_length):
    """Create sequences for LSTM input"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    
    return np.array(X_seq), np.array(y_seq)

def train_lstm_model(features_df, target_column='close', forecast_days=5, test_size=0.2, seq_length=10):
    """
    Train an LSTM model for price prediction
    
    Args:
        features_df: DataFrame with engineered features
        target_column: Column to predict
        forecast_days: Number of days ahead to predict
        test_size: Proportion of data for testing
        seq_length: Length of input sequences for LSTM
        
    Returns:
        Dictionary with model, scalers, and evaluation results
    """
    print("\nTraining LSTM model...")
    
    # Create target variable (shifted price)
    df = features_df.copy()
    df['target'] = df[target_column].shift(-forecast_days)
    
    # Drop rows with NaN targets
    df = df.dropna()
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Use MinMaxScaler instead of StandardScaler
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    # Store feature names for later use
    feature_scaler.feature_names = X.columns.tolist()
    
    # Create sequences for LSTM
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)
    
    print(f"LSTM sequence shape: {X_seq.shape}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=test_size, shuffle=False
    )
    
    # Define LSTM architecture
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, 
             input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    
    # Train the model with early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate the model
    y_pred = model.predict(X_test).flatten()
    
    # Calculate metrics on scaled data
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Convert predictions back to original scale for interpretable metrics
    y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    mse_orig = mean_squared_error(y_test_orig, y_pred_orig)
    rmse_orig = np.sqrt(mse_orig)
    mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
    r2 = r2_score(y_test_orig, y_pred_orig)
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100
    
    # Print evaluation metrics
    print(f"LSTM Model Metrics:")
    print(f"  MSE: {mse_orig:.4f}")
    print(f"  RMSE: {rmse_orig:.4f}")
    print(f"  MAE: {mae_orig:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # Results dataframe
    results_df = pd.DataFrame({
        'actual': y_test_orig,
        'predicted': y_pred_orig,
        'error': y_test_orig - y_pred_orig,
        'error_pct': ((y_test_orig - y_pred_orig) / y_test_orig) * 100
    })
    
    return {
        'model': model,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'seq_length': seq_length,
        'feature_names': X.columns.tolist(),
        'metrics': {
            'mse': mse_orig,
            'rmse': rmse_orig,
            'mae': mae_orig,
            'r2': r2,
            'mape': mape
        },
        'results': results_df,
        'history': history.history
    }

def predict_with_lstm(model_results, features_df, days=5):
    """
    Make predictions using the trained LSTM model
    
    Args:
        model_results: Dictionary with trained LSTM model and scalers
        features_df: DataFrame with the most recent data
        days: Number of days to predict
        
    Returns:
        DataFrame with predictions
    """
    # Extract model components
    model = model_results['model']
    feature_scaler = model_results['feature_scaler']
    target_scaler = model_results['target_scaler']
    seq_length = model_results['seq_length']
    feature_names = model_results['feature_names']
    
    # Ensure we have the right features
    features = features_df[feature_names].copy()
    
    # Get the most recent sequence
    latest_data = features.iloc[-seq_length:].copy()
    
    # Scale the features
    scaled_data = feature_scaler.transform(latest_data)
    
    # Store predictions
    predictions = []
    dates = []
    last_date = features.index[-1]
    
    # Create a copy of the scaled data to update during prediction
    prediction_sequence = scaled_data.copy()
    
    # Make predictions for each future day
    for i in range(days):
        # Reshape for LSTM input [samples, time steps, features]
        seq_reshape = prediction_sequence.reshape(1, seq_length, len(feature_names))
        
        # Predict
        scaled_pred = model.predict(seq_reshape)[0][0]
        
        # Convert back to original scale
        prediction = target_scaler.inverse_transform([[scaled_pred]])[0][0]
        
        # Calculate next date
        future_date = last_date + pd.Timedelta(days=i+1)
        dates.append(future_date)
        predictions.append(prediction)
        
        # Update the sequence for next prediction:
        # First, create a new row based on the last available data
        new_features = features.iloc[-1].copy()
        
        # Update price-related columns with the prediction - Remove random noise
        new_features['close'] = prediction
        prev_close = features.iloc[-1]['close']
        new_features['open'] = prev_close * 1.001 if prediction > prev_close else prev_close * 0.999
        new_features['high'] = max(prediction, new_features['open']) * 1.01
        new_features['low'] = min(prediction, new_features['open']) * 0.99
        
        # Ensure high >= open/close and low <= open/close
        new_features['high'] = max(new_features['high'], new_features['close'], new_features['open'])
        new_features['low'] = min(new_features['low'], new_features['close'], new_features['open'])

        # Update technical indicators based on the new predicted price
        if 'ma5' in feature_names:
            ma5_values = list(features.iloc[-4:]['close'].values) + [prediction]
            new_features['ma5'] = sum(ma5_values) / 5
            
        if 'ma7' in feature_names:
            ma7_values = list(features.iloc[-6:]['close'].values) + [prediction]
            new_features['ma7'] = sum(ma7_values) / min(7, len(ma7_values))
            
        if 'ma14' in feature_names:
            ma14_values = list(features.iloc[-13:]['close'].values) + [prediction]
            new_features['ma14'] = sum(ma14_values) / min(14, len(ma14_values))
            
        if 'ema5' in feature_names:
            alpha = 2 / (5 + 1)
            new_features['ema5'] = (prediction * alpha) + (features.iloc[-1]['ema5'] * (1 - alpha))
            
        if 'ema14' in feature_names:
            alpha = 2 / (14 + 1)
            new_features['ema14'] = (prediction * alpha) + (features.iloc[-1]['ema14'] * (1 - alpha))
        
        if 'price_change_1d' in feature_names:
            new_features['price_change_1d'] = (prediction / features.iloc[-1]['close']) - 1
            
        if 'price_change_3d' in feature_names:
            if len(features) >= 3:
                new_features['price_change_3d'] = (prediction / features.iloc[-3]['close']) - 1
        
        if 'price_change_7d' in feature_names:
            if len(features) >= 7:
                new_features['price_change_7d'] = (prediction / features.iloc[-7]['close']) - 1
                
        if 'volatility_5d' in feature_names:
            vol_values = list(features.iloc[-4:]['close'].values) + [prediction]
            new_features['volatility_5d'] = np.std(vol_values)
            
        if 'volatility_ratio' in feature_names:
            vol_values = list(features.iloc[-4:]['close'].values) + [prediction]
            volatility = np.std(vol_values)
            new_features['volatility_ratio'] = volatility / prediction if prediction != 0 else 0
            
        if 'ma_crossover' in feature_names and 'ma5' in new_features and 'ma14' in new_features:
            new_features['ma_crossover'] = int(new_features['ma5'] > new_features['ma14'])
            
        if 'bb_upper' in feature_names and 'ma14' in new_features:
            std_20 = np.std(list(features.iloc[-19:]['close'].values) + [prediction])
            new_features['bb_upper'] = new_features['ma14'] + (std_20 * 2)
            
        if 'bb_lower' in feature_names and 'ma14' in new_features:
            std_20 = np.std(list(features.iloc[-19:]['close'].values) + [prediction])
            new_features['bb_lower'] = new_features['ma14'] - (std_20 * 2)
            
        if 'bb_width' in feature_names and 'bb_upper' in new_features and 'bb_lower' in new_features and 'ma14' in new_features:
            new_features['bb_width'] = (new_features['bb_upper'] - new_features['bb_lower']) / new_features['ma14'] if new_features['ma14'] != 0 else 0
            
        if 'rsi' in feature_names and len(features) >= 14:
            deltas = list(features.iloc[-14:]['close'].diff().dropna().values)
            deltas.append(prediction - features.iloc[-1]['close'])
            gains = [max(d, 0) for d in deltas]
            losses = [abs(min(d, 0)) for d in deltas]
            avg_gain = sum(gains) / 14
            avg_loss = sum(losses) / 14
            if avg_loss == 0:
                new_features['rsi'] = 100
            else:
                rs = avg_gain / avg_loss
                new_features['rsi'] = 100 - (100 / (1 + rs))
                
        if 'momentum' in feature_names:
            new_features['momentum'] = prediction - features.iloc[-5]['close'] if len(features) >= 5 else 0
        
        # Carry forward other features like fear_greed_index and sentiment metrics
        for col in feature_names:
            if col not in new_features or pd.isna(new_features[col]):
                new_features[col] = features.iloc[-1][col]
        
        # Scale the new row - Ensure feature names are used
        new_scaled = feature_scaler.transform(pd.DataFrame([new_features.values], columns=feature_names))[0]
        
        # Remove first row and append the new one to the sequence
        prediction_sequence = np.vstack([prediction_sequence[1:], new_scaled])
        
        # Add the new features to our features dataframe for the next iteration
        features.loc[future_date] = new_features
    
    # Create prediction DataFrame
    prediction_df = pd.DataFrame({
        'date': dates,
        'predicted_price': predictions
    })
    prediction_df.set_index('date', inplace=True)
    
    return prediction_df

def backtest_lstm_model(model_results, features_df, lookback_days=30, prediction_days=5):
    """
    Backtest the LSTM model on historical data.

    Args:
        model_results: Dictionary containing the trained LSTM model, scalers, etc.
        features_df: DataFrame with historical features.
        lookback_days: Number of days of history to use for each prediction point.
                       Should be >= seq_length.
        prediction_days: Number of days ahead to predict at each point.

    Returns:
        Dictionary with backtest results and metrics.
    """
    print("Backtesting LSTM model...")

    seq_length = model_results['seq_length']
    target_column = 'close' # Assuming 'close' is the target

    # Ensure lookback includes sequence length
    effective_lookback = max(lookback_days, seq_length)

    # Ensure we have enough data
    if len(features_df) < effective_lookback + prediction_days + 20: # Add buffer
        print("Not enough data for LSTM backtesting")
        return {
            'results': pd.DataFrame(),
            'metrics': {'mse': float('nan'), 'rmse': float('nan'), 'mae': float('nan'), 'mape': float('nan')}
        }

    # Determine backtest points (e.g., every 10 days)
    # Start index ensures we have enough data for the first sequence + lookback
    start_index = effective_lookback
    backtest_points = list(range(start_index, len(features_df) - prediction_days, 10))

    results = []

    for i in backtest_points:
        # Get historical data up to this point (including lookback for feature calculation)
        historical_data = features_df.iloc[:i].copy()

        # Get actual future data for comparison
        actual_future = features_df.iloc[i : i + prediction_days].copy()

        if len(actual_future) < prediction_days:
            continue # Skip if not enough future data

        # Make prediction using the LSTM logic
        # Use historical_data which includes enough points for the initial sequence
        try:
            future_pred_df = predict_with_lstm(
                model_results,
                historical_data, # Pass the historical segment
                days=prediction_days
            )
        except Exception as e:
             print(f"Error during LSTM prediction at index {i}: {e}")
             continue # Skip this point if prediction fails


        # Compare prediction with actual data
        for j in range(min(len(future_pred_df), len(actual_future))):
            pred_date = future_pred_df.index[j]
            # Find the closest date in actual data (usually the same index)
            actual_date = actual_future.index[j]

            pred_price = future_pred_df['predicted_price'].iloc[j]
            actual_price = actual_future[target_column].loc[actual_date]

            error = actual_price - pred_price
            error_pct = (error / actual_price) * 100 if actual_price != 0 else 0

            results.append({
                'prediction_date': historical_data.index[-1], # Date prediction was made
                'target_date': pred_date,
                'predicted_price': pred_price,
                'actual_price': actual_price,
                'error': error,
                'error_pct': error_pct
            })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Check if we have results before calculating metrics
    if results_df.empty:
        print("No valid LSTM backtest results generated")
        return {
            'results': pd.DataFrame(),
            'metrics': {'mse': float('nan'), 'rmse': float('nan'), 'mae': float('nan'), 'mape': float('nan')}
        }

    # Calculate aggregate metrics
    actual = results_df['actual_price']
    predicted = results_df['predicted_price']

    # Ensure no NaNs before calculating metrics
    valid_idx = actual.notna() & predicted.notna()
    if not valid_idx.all():
        print(f"Warning: Found NaNs in backtest results. Dropping {len(results_df) - valid_idx.sum()} rows.")
        actual = actual[valid_idx]
        predicted = predicted[valid_idx]
        results_df = results_df[valid_idx]

    if actual.empty:
         print("No valid non-NaN backtest results after filtering.")
         return {
            'results': pd.DataFrame(),
            'metrics': {'mse': float('nan'), 'rmse': float('nan'), 'mae': float('nan'), 'mape': float('nan')}
        }


    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)

    # Calculate MAPE safely
    valid_actual = actual[actual != 0]
    if not valid_actual.empty:
        mape = np.mean(np.abs((actual[actual != 0] - predicted[actual != 0]) / actual[actual != 0])) * 100
    else:
        mape = float('nan')

    print(f"LSTM Backtest Results:")
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