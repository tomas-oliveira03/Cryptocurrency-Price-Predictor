import os
import pandas as pd
import numpy as np
from bson import CodecOptions
from pymongo import MongoClient
import json
import random
import tensorflow as tf
import fetchData
import preProcessor
import engineFeatures
from visualizePredictions import visualize_predictions
import lstmModel 

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Seeds set to {seed} for random, numpy, and tensorflow (if available).")

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
        

    def runEverything(self, cryptoCoin, forcastDays, initialFetchDays):
        set_seeds(42)
        
        # Save raw feature data first
        dataDir = "Services/Models/data"
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)


        # Step 1: Get raw data for prediction
        rawData = fetchData.getDataForPrediction(self, cryptoCoin=cryptoCoin, numberOfPastDaysOfData=initialFetchDays)
        
        # Step 2: Preprocess the data
        processedData = preProcessor.preprocessData(self, rawData)
        
        # Step 3: Engineer features
        featuresDF = engineFeatures.engineFeatures(self, processedData)
        
        # Step 4: Train LSTM model using all data except the last 7 days for testing
        print("\n--- Training LSTM Model (Testing on Last 7 Days) ---")
        
        # Use a smaller sequence length if dataset is smaller
        seq_length = min(10, len(featuresDF) // 5)  # Adjust sequence length based on data size
        if seq_length < 1: seq_length = 1 # Ensure seq_length is at least 1
        
        test_days_for_training = 7 # Use last 7 days for testing during training

        # Check if we have enough data for the test split + sequence length
        min_data_needed = test_days_for_training + seq_length + 1 # +1 for the target shift
        if len(featuresDF) < min_data_needed:
             raise ValueError(f"Not enough historical data ({len(featuresDF)} days) to train and test with seq_length={seq_length} and test_days={test_days_for_training}. Need at least {min_data_needed} days.")

        lstm_results = lstmModel.trainLstmModel(
            featuresDF,
            target_column='close',
            forecast_days=1, # Typically train LSTM for 1-step ahead prediction
            test_days=test_days_for_training,   # Use last 7 days for testing
            seq_length=seq_length
        )
            
        # Step 5: Predict Future with the Trained LSTM Model
        print(f"\n--- Predicting Next {forcastDays} Days using LSTM Model ---")

        future_predictions = lstmModel.predictWithLstm(
            lstm_results, # Use the model trained in the previous step
            featuresDF,
            days=forcastDays
        )

        # Calculate overall prediction trend
        overall_change_pct = 0.0
        if not future_predictions.empty and not featuresDF.empty:
            last_hist_price = featuresDF['close'].iloc[-1]
            last_pred_price = future_predictions['predicted_price'].iloc[-1]
            if last_hist_price != 0:
                overall_change_pct = ((last_pred_price - last_hist_price) / last_hist_price) * 100
            else:
                overall_change_pct = float('inf') if last_pred_price > 0 else 0.0 # Handle division by zero

        # Visualize final predictions
        print("\nCreating final prediction visualization...")
        sentiment_col = 'pct_positive' if 'pct_positive' in featuresDF.columns else None
        sentiment_plot_data = featuresDF[[sentiment_col]] if sentiment_col and sentiment_col in featuresDF.columns else None
        
        visualize_predictions(
            featuresDF[['close']],
            future_predictions,
            sentiment_data=sentiment_plot_data,
            save_path=f"{dataDir}/crypto_price_prediction_lstm_{forcastDays}d.png"
        )

        # Save predictions to CSV
        predictions_csv = f"{dataDir}/price_predictions_lstm_{forcastDays}d.csv"
        future_predictions.to_csv(predictions_csv)
        print(f"Final predictions saved to {predictions_csv}")

        # Generate and save JSON data export
        print("\nGenerating JSON data export...")
        json_export_path = os.path.join(dataDir, f"{cryptoCoin}_data_export_lstm_{forcastDays}d.json")

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
            "coin": cryptoCoin,
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
            "lstm_results": lstm_results, # This now holds the final trained model and its metrics
            "predictions": future_predictions,
            "json_export_path": json_export_path,
            "forcastDays": forcastDays,
            "overall_change_pct": overall_change_pct # Add the calculated percentage change
        }
            
        return results_dict


if __name__ == "__main__":
    try:
        predictionModel = PredictionModel()
        
        # --- Configuration ---
        cryptoCoin = "BTC"
        forcastDays = 7   # Days to predict into the future
        initialFetchDays = 365 * 2 # Fetch ample history initially (e.g., 2 years)
        # -------------------
        
        results = predictionModel.runEverything(cryptoCoin, forcastDays, initialFetchDays)
        
        print("\n--- Final Results Summary ---")

        # Access config values from the results dictionary
        fc_days = results.get('forcastDays', 'N/A')
        overall_change = results.get('overall_change_pct', None)

        print(f"Model used: LSTM")
        print("LSTM Model Metrics (Tested on Last 7 Days):") # Updated description
        lstm_metrics = results.get('lstm_results', {}).get('metrics', {})
        for metric, value in lstm_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")

        print(f"Predictions saved to CSV and JSON: {results.get('json_export_path')}")
        print(f"Final {fc_days}-day Predictions:")
        print(results.get('predictions', pd.DataFrame()))

        # Print the overall trend
        if overall_change is not None:
            direction = "increase" if overall_change >= 0 else "decrease"
            if overall_change == float('inf'):
                 print(f"\nOverall Trend: Price predicted to increase significantly from zero.")
            else:
                 print(f"\nOverall Trend: {abs(overall_change):.2f}% {direction} from last known price over the next {fc_days} days.")
        else:
            print("\nOverall Trend: Could not be calculated.")

    except ValueError as ve:
        print(f"\n--- Execution Failed (ValueError) ---")
        print(ve)
    except Exception as e:
        print(f"\n--- Execution Failed (Unexpected Error) ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
