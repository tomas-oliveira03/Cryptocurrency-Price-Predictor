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

def train_lstm_model(X_train_seq, y_train_seq, X_test_seq, y_test_seq, target_scaler, seq_length, n_features):
    """
    Builds, trains, and evaluates an LSTM model.

    Args:
        X_train_seq, y_train_seq: Training sequences.
        X_test_seq, y_test_seq: Testing sequences.
        target_scaler: Fitted scaler for the target variable.
        seq_length: Sequence length used.
        n_features: Number of input features.

    Returns:
        Dictionary containing the trained model, metrics (train, val, test), and history.
        Returns None if training fails or not enough data.
    """
    print("Building LSTM model...")
    if len(X_train_seq) < 10 or len(X_test_seq) < 1: # Need minimum data
        print("Not enough sequence data for LSTM training.")
        return None

    # Simplified LSTM architecture
    units = 50
    dropout_rate = 0.2
    epochs = 100
    batch_size = min(32, len(X_train_seq) // 2 if len(X_train_seq) >= 2 else 1)

    model = Sequential([
        LSTM(units, activation='relu', return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(dropout_rate),
        LSTM(units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print(f"Training LSTM with {len(X_train_seq)} samples...")
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1, # Use part of training data for validation
        callbacks=[early_stopping],
        verbose=1 # Set to 1 or 2 for progress, 0 for silent
    )

    # --- Evaluate on Test Set ---
    print("Evaluating LSTM on test set...")
    y_pred_scaled = model.predict(X_test_seq).flatten()

    # Inverse transform to original scale
    y_test_orig = target_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
    y_pred_orig = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # Calculate Test Metrics
    test_mse = mean_squared_error(y_test_orig, y_pred_orig)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test_orig, y_pred_orig)
    test_r2 = r2_score(y_test_orig, y_pred_orig)
    test_mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100 if np.all(y_test_orig != 0) else np.inf

    test_metrics = {'mse': test_mse, 'rmse': test_rmse, 'mae': test_mae, 'r2': test_r2, 'mape': test_mape}

    # --- Get Training and Validation Metrics from History ---
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    train_mae_scaled = history.history.get('mae', [np.nan])[-1]
    val_mae_scaled = history.history.get('val_mae', [np.nan])[-1]

    train_metrics = {'scaled_loss': train_loss, 'scaled_mae': train_mae_scaled}
    validation_metrics = {'scaled_loss': val_loss, 'scaled_mae': val_mae_scaled}

    print("LSTM Training and Evaluation Complete.")
    return {
        'model': model,
        'test_metrics': test_metrics,
        'train_metrics': train_metrics, # Scaled metrics from history
        'validation_metrics': validation_metrics, # Scaled metrics from history
        'history': history.history,
        'predictions_test': y_pred_orig, # Store original scale predictions
        'actual_test': y_test_orig      # Store original scale actuals
    }

def predict_with_lstm(lstm_results, historical_data, days=5):
    """
    Make future predictions using a trained LSTM model.

    Args:
        lstm_results (dict): Dictionary containing the trained LSTM model,
                             scalers ('feature', 'target'), and seq_length.
        historical_data (pd.DataFrame): DataFrame with historical features.
        days (int): Number of future days to predict.

    Returns:
        pd.DataFrame: DataFrame with future dates and predicted prices.
    """
    model = lstm_results['model']
    feature_scaler = lstm_results['feature_scaler']
    target_scaler = lstm_results['target_scaler']
    seq_length = lstm_results['seq_length']

    # Get feature names from the scaler
    if hasattr(feature_scaler, 'feature_names_in_'):
        features = feature_scaler.feature_names_in_
    elif hasattr(feature_scaler, 'feature_names'): # Fallback if using older sklearn?
         features = feature_scaler.feature_names
    else:
        # If scaler has no feature names, try getting from historical_data, excluding 'target' if present
        features = [col for col in historical_data.columns if col != 'target']
        print("Warning: Feature names not found on scaler, inferring from historical_data.")

    # Use the last 'seq_length' days from historical data as the starting sequence
    last_sequence_df = historical_data.iloc[-seq_length:][features]

    # Ensure columns match the order expected by the scaler
    last_sequence_df = last_sequence_df[features]

    current_sequence_scaled = feature_scaler.transform(last_sequence_df)

    predictions_scaled = []

    for _ in range(days):
        # Reshape sequence for prediction: (1, seq_length, n_features)
        input_sequence = current_sequence_scaled.reshape(1, seq_length, len(features))

        # Predict the next step (scaled)
        next_pred_scaled = model.predict(input_sequence)[0][0]
        predictions_scaled.append(next_pred_scaled)

        # Update the sequence: drop the first step, append the prediction
        try:
            # Find index of 'close' or similar target base column if possible
            target_base_col = 'close' # Assume 'close' was the base for the target
            target_idx = list(features).index(target_base_col)
            new_step_scaled = current_sequence_scaled[-1].copy()
            new_step_placeholder = np.append(current_sequence_scaled[-1, 1:], next_pred_scaled)

            # Roll the sequence and append the new step placeholder
            current_sequence_scaled = np.vstack((current_sequence_scaled[1:], new_step_placeholder.reshape(1, -1)))

        except ValueError:
             print("Warning: Cannot find target base column ('close') in features for sequence update.")
             current_sequence_scaled = np.vstack((current_sequence_scaled[1:], current_sequence_scaled[-1].reshape(1, -1)))

    # Inverse transform the predictions
    predictions = target_scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()

    # Create future dates
    last_date = historical_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)

    # Create prediction DataFrame
    prediction_df = pd.DataFrame({'predicted_price': predictions}, index=future_dates)
    prediction_df.index.name = 'date'

    return prediction_df
