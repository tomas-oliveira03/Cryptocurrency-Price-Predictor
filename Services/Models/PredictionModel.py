import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from bson import CodecOptions
from pymongo import MongoClient
import json
import random  # Import random

# Import TensorFlow if LSTM uses it
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not found. LSTM reproducibility might be affected if it uses TensorFlow.")

import fetchData
import preProcessor
import engineFeatures
from simplePredictionModel import visualize_predictions
import lstmModel  # Import the LSTM module

# --- Seed Setting Function ---
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if TENSORFLOW_AVAILABLE:
        tf.random.set_seed(seed)
    print(f"Seeds set to {seed} for random, numpy, and tensorflow (if available).")
# ---------------------------

class PredictionModel:
    def __init__(self):
        
        mongoDBURI = os.getenv("MONGODB_URI")
        if not mongoDBURI:
            raise ValueError("Please set the MONGODB_URI environment variable first.")

        # Database connection
        mongoClient = MongoClient(mongoDBURI)
        
        self.cryptoFearGreedDB = mongoClient['ASM'].get_collection('crypto-fear-greed', codec_options=CodecOptions(tz_aware=True))
        self.detailedCryptoData = mongoClient['ASM'].get_collection('detailed-crypto-data', codec_options=CodecOptions(tz_aware=True))
        
        self.redditDB = mongoClient['ASM'].get_collection('reddit', codec_options=CodecOptions(tz_aware=True))
        self.forumDB = mongoClient['ASM'].get_collection('forum', codec_options=CodecOptions(tz_aware=True))
        self.articlesDB = mongoClient['ASM'].get_collection('articles', codec_options=CodecOptions(tz_aware=True))
        

    def runEverything(self):
        # --- Set Seeds for Reproducibility ---
        set_seeds(42)
        # ------------------------------------

        # --- Configuration ---
        crypto_coin_symbol = "BTC"
        forecast_days = 7   # Days to predict into the future
        initial_fetch_days = 365 * 2 # Fetch ample history initially (e.g., 2 years)
        # -------------------

        # Step 1: Get raw data for prediction
        rawData = fetchData.getDataForPrediction(self, cryptoCoin=crypto_coin_symbol, numberOfPastDaysOfData=initial_fetch_days)
        
        # Step 2: Preprocess the data
        processedData = preProcessor.preprocessData(self, rawData)
        
        # Step 3: Engineer features
        featuresDF = engineFeatures.engineFeatures(self, processedData)
        
        # Print feature summary
        print("\nEngineered Features Summary:")
        print(f"Total features: {len(featuresDF.columns)}")
        print(f"Total data points: {len(featuresDF)}")
        print("Feature names:", ", ".join(featuresDF.columns.tolist()[:5]) + "...")
        
        # Save raw feature data first
        csv_dir = "Services/Models/data"
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        
        features_csv = f"{csv_dir}/raw_features.csv"
        featuresDF.to_csv(features_csv)
        print(f"Raw features saved to {features_csv}")
        
        # Step 4: Train LSTM model (Initial training)
        print("\n--- Training Initial LSTM Model ---")
        # --- Set seeds again before LSTM training ---
        set_seeds(42)
        # ------------------------------------------
        
        # Use a smaller sequence length if dataset is smaller
        seq_length = min(10, len(featuresDF) // 5)  # Adjust sequence length based on data size
        if seq_length < 1: seq_length = 1 # Ensure seq_length is at least 1

        lstm_results = lstmModel.train_lstm_model(
            featuresDF,
            target_column='close',
            forecast_days=1, # Typically train LSTM for 1-step ahead prediction
            test_size=0.2,   # Use a portion for validation during training
            seq_length=seq_length
        )
        
        print("\nLSTM Model Initial Training Metrics:")
        for metric, value in lstm_results['metrics'].items():
            print(f"  {metric.upper()}: {value:.4f}")
            
        # Step 5: Retrain LSTM on full dataset for final model
        print("\n--- Retraining LSTM Model on Full Historical Data ---")
        # --- Set seeds again before final LSTM training ---
        set_seeds(42)
        # ------------------------------------------------
        
        # Retrain LSTM on the entire dataset
        retrain_seq_length = min(10, len(featuresDF) // 10)
        if retrain_seq_length < 1: retrain_seq_length = 1

        final_lstm_model = lstmModel.train_lstm_model(
            featuresDF, # Use full dataset
            target_column='close',
            forecast_days=1,  # Still 1-step ahead prediction
            test_size=0.01,   # Minimal test set for final training
            seq_length=retrain_seq_length
        )
        print("LSTM retrained successfully on full data.")

        # Step 6: Predict Future with LSTM
        print(f"\n--- Predicting Next {forecast_days} Days using LSTM Model ---")

        future_predictions = lstmModel.predict_with_lstm(
            final_lstm_model,
            featuresDF,
            days=forecast_days
        )

        # Step 7: Output Results
        # Print detailed prediction information
        print("\nDetailed Future Price Predictions:")
        if not future_predictions.empty:
            for date, row in future_predictions.iterrows():
                print(f"  {date.strftime('%Y-%m-%d')}: ${row['predicted_price']:.2f}")
            
            # Calculate overall prediction trend
            first_pred = future_predictions['predicted_price'].iloc[0]
            last_pred = future_predictions['predicted_price'].iloc[-1]
            total_change_pct = ((last_pred - first_pred) / first_pred) * 100 if first_pred != 0 else 0
            print(f"\nOverall {forecast_days}-day prediction trend: {total_change_pct:+.2f}%")
        else:
            print("  No future predictions were generated.")

        # Visualize final predictions
        print("\nCreating final prediction visualization...")
        sentiment_col = 'pct_positive' if 'pct_positive' in featuresDF.columns else None
        sentiment_plot_data = featuresDF[[sentiment_col]] if sentiment_col and sentiment_col in featuresDF.columns else None
        
        visualize_predictions(
            featuresDF[['close']],
            future_predictions,
            sentiment_data=sentiment_plot_data,
            save_path=f"{csv_dir}/crypto_price_prediction_lstm_{forecast_days}d.png"
        )

        # Save predictions to CSV
        predictions_csv = f"{csv_dir}/price_predictions_lstm_{forecast_days}d.csv"
        future_predictions.to_csv(predictions_csv)
        print(f"Final predictions saved to {predictions_csv}")

        # Generate and save JSON data export
        print("\nGenerating JSON data export...")
        json_export_path = os.path.join(csv_dir, f"{crypto_coin_symbol}_data_export_lstm_{forecast_days}d.json")

        historical_price_data = [
            {"date": date.strftime('%Y-%m-%d'), "price": price}
            for date, price in featuresDF['close'].items()
        ]

        predicted_price_data = [
            {"date": date.strftime('%Y-%m-%d'), "price": price}
            for date, price in future_predictions['predicted_price'].items()
        ]

        positive_sentiment_data = []
        if 'pct_positive' in featuresDF.columns:
            positive_sentiment_data = [
                {"date": date.strftime('%Y-%m-%d'), "sentiment": sentiment}
                for date, sentiment in featuresDF['pct_positive'].items() if pd.notna(sentiment)
            ]

        json_data = {
            "coin": crypto_coin_symbol,
            "historical_price": historical_price_data,
            "predicted_price": predicted_price_data,
            "positive_sentiment_ratio": positive_sentiment_data
        }

        with open(json_export_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON data export saved to {json_export_path}")

        # Return results dictionary
        results_dict = {
            "features": featuresDF,
            "lstm_results": final_lstm_model,
            "predictions": future_predictions,
            "json_export_path": json_export_path,
            "forecast_days": forecast_days
        }
            
        return results_dict

if __name__ == "__main__":
    # Example usage
    try:
        predictionModel = PredictionModel()
        results = predictionModel.runEverything()
        print("\n--- Final Results Summary ---")

        # Access config values from the results dictionary
        fc_days = results.get('forecast_days', 'N/A')

        print(f"Model used: LSTM")
        print("LSTM Model Metrics:")
        lstm_metrics = results.get('lstm_results', {}).get('metrics', {})
        for metric, value in lstm_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")

        print(f"Predictions saved to CSV and JSON: {results.get('json_export_path')}")
        print(f"Final {fc_days}-day Predictions:")
        print(results.get('predictions', pd.DataFrame()))
    except ValueError as ve:
        print(f"\n--- Execution Failed (ValueError) ---")
        print(ve)
    except Exception as e:
        print(f"\n--- Execution Failed (Unexpected Error) ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
