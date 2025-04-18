import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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