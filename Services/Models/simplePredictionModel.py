import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lstmModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM

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
    df = features_df.copy()
    df['target'] = df[target_column].shift(-forecast_days)
    df = df.dropna()
    return df

def train_model(features_df, target_column='close', forecast_days=5, test_size=0.2, lstm_seq_length=10):
    """
    Train and evaluate an LSTM model.

    Args:
        features_df: DataFrame with features and target.
        target_column: Name of the target column (e.g., 'close').
        forecast_days: Number of days ahead the target represents.
        test_size: Proportion of data for the test set.
        lstm_seq_length: Sequence length for LSTM model.

    Returns:
        Dictionary containing the LSTM model, scalers, features, sequence length,
        and metrics (train, val, test). Returns None if training fails.
    """
    print("Starting LSTM model training...")
    df = features_df.copy()
    df['target'] = df[target_column].shift(-forecast_days)
    df.dropna(inplace=True)

    if df.empty:
        print("Not enough data after creating target variable.")
        return None

    features = df.drop('target', axis=1)
    target = df['target']
    feature_names = features.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, shuffle=False
    )
    print(f"Training data: {len(X_train)} samples, Test data: {len(X_test)} samples")

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    feature_scaler.feature_names_in_ = np.array(feature_names, dtype=object)
    feature_scaler.n_features_in_ = len(feature_names)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

    print("\n--- Training LSTM Model ---")
    lstm_results = None
    if len(X_train) > lstm_seq_length + 5:
        try:
            X_train_seq, y_train_seq = lstmModel.create_sequences(X_train_scaled, y_train_scaled, lstm_seq_length)
            X_test_seq, y_test_seq = lstmModel.create_sequences(X_test_scaled, y_test_scaled, lstm_seq_length)

            lstm_run_results = lstmModel.train_lstm_model(
                X_train_seq, y_train_seq, X_test_seq, y_test_seq,
                target_scaler,
                lstm_seq_length,
                n_features=X_train_scaled.shape[1]
            )

            if lstm_run_results:
                lstm_results = {
                    'model': lstm_run_results['model'],
                    'scaler': {'feature': feature_scaler, 'target': target_scaler},
                    'features': feature_names,
                    'seq_length': lstm_seq_length,
                    'metrics': lstm_run_results['test_metrics'],
                    'train_metrics': lstm_run_results['train_metrics'],
                    'validation_metrics': lstm_run_results['validation_metrics'],
                    'predictions_test': lstm_run_results.get('predictions_test'),
                    'actual_test': lstm_run_results.get('actual_test')
                }
                print(f"LSTM Training Complete. Test RMSE: {lstm_results['metrics']['rmse']:.4f}")
            else:
                print("LSTM training skipped or failed.")
                return None

        except Exception as e:
            print(f"Error during LSTM training: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print(f"Skipping LSTM: Not enough training data ({len(X_train)}) for sequence length {lstm_seq_length}.")
        return None

    print("LSTM model training finished.")
    return lstm_results

def visualize_predictions(historical_data, predictions, save_path="price_prediction.png"):
    """
    Visualize historical data and predictions
    
    Args:
        historical_data: DataFrame with historical data (needs at least 'close' column)
        predictions: DataFrame with predictions (needs 'predicted_price' column)
        save_path: Path to save the visualization
    """
    last_n_days = 30
    if len(historical_data) > last_n_days:
        historical_subset = historical_data.iloc[-last_n_days:]
    else:
        historical_subset = historical_data

    plt.figure(figsize=(12, 6))
    plt.plot(historical_subset.index, historical_subset['close'], 
             label='Historical Price', color='blue', linewidth=2)
    plt.plot(predictions.index, predictions['predicted_price'], 
             label='Predicted Price', color='red', linestyle='--', linewidth=2)
    plt.scatter(historical_subset.index, historical_subset['close'], color='blue', s=30)
    plt.scatter(predictions.index, predictions['predicted_price'], color='red', s=30)
    last_hist_date = historical_data.index[-1]
    last_hist_price = historical_data['close'].iloc[-1]
    plt.axvline(x=last_hist_date, color='gray', linestyle=':', linewidth=1)
    last_pred_price = predictions['predicted_price'].iloc[-1]
    change_pct = ((last_pred_price - last_hist_price) / last_hist_price) * 100
    direction = "increase" if change_pct >= 0 else "decrease"
    plt.title(f'Price Prediction: {abs(change_pct):.2f}% {direction} from last known price over next {len(predictions)} days', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Prediction visualization saved as {save_path}")

def backtest_model(model, scaler, features, historical_data, seq_length):
    """
    Backtest the LSTM model on historical data using a rolling window approach.

    Args:
        model: Trained LSTM model object.
        scaler: Fitted feature scaler object (MinMaxScaler).
        features (list): List of feature names used by the model.
        historical_data (pd.DataFrame): DataFrame with historical features.
        seq_length (int): Sequence length required for LSTM model.

    Returns:
        pd.DataFrame: DataFrame with backtest predictions and actual values.
    """
    print("Backtesting LSTM model...")
    backtest_days = 30
    if len(historical_data) < backtest_days + seq_length + 5:
        print("Not enough historical data for backtesting. Skipping.")
        return pd.DataFrame()

    predictions = []
    actuals = []
    dates = []

    if not hasattr(scaler, 'feature_names_in_'):
        if hasattr(scaler, 'feature_names'):
             scaler.feature_names_in_ = np.array(scaler.feature_names, dtype=object)
             scaler.n_features_in_ = len(scaler.feature_names)
        else:
             print("Warning: Scaler missing feature names for backtesting.")
             scaler.feature_names_in_ = np.array(features, dtype=object)
             scaler.n_features_in_ = len(features)

    for i in range(len(historical_data) - backtest_days, len(historical_data)):
        current_date = historical_data.index[i]
        actual_value = historical_data['close'].iloc[i]
        start_idx = i - seq_length
        end_idx = i
        if start_idx < 0:
            continue

        input_sequence_df = historical_data.iloc[start_idx:end_idx][features]

        try:
            input_sequence_df = input_sequence_df[scaler.feature_names_in_]
        except KeyError as e:
            print(f"KeyError during backtest scaling: {e}. Check feature consistency.")
            print(f"Expected features: {scaler.feature_names_in_}")
            print(f"Available features: {input_sequence_df.columns.tolist()}")
            continue

        input_sequence_scaled = scaler.transform(input_sequence_df)
        input_sequence_reshaped = input_sequence_scaled.reshape(1, seq_length, len(features))
        pred_scaled = model.predict(input_sequence_reshaped)[0][0]
        temp_target_scaler = MinMaxScaler(feature_range=(0, 1))
        dummy_prices = np.linspace(historical_data['close'].min(), historical_data['close'].max(), 100).reshape(-1, 1)
        temp_target_scaler.fit(dummy_prices)
        prediction = temp_target_scaler.inverse_transform([[pred_scaled]])[0][0]

        predictions.append(prediction)
        actuals.append(actual_value)
        dates.append(current_date)

    results_df = pd.DataFrame({'date': dates, 'predicted': predictions, 'actual': actuals})
    results_df.set_index('date', inplace=True)

    if not results_df.empty:
        mse = mean_squared_error(results_df['actual'], results_df['predicted'])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(results_df['actual'], results_df['predicted'])
        mape = np.mean(np.abs((results_df['actual'] - results_df['predicted']) / results_df['actual'])) * 100
        print(f"Backtest Results (RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%)")
    else:
        print("Backtest did not produce results.")

    return results_df
