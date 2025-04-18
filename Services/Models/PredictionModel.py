import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from bson import CodecOptions
from pymongo import MongoClient

import fetchData
import preProcessor
import engineFeatures
import simplePredictionModel


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
        # Step 1: Get raw data
        rawData = fetchData.getDataForPrediction(self, cryptoCoin="BTC", numberOfPastDaysOfData=365)

        # Step 2: Preprocess data
        processedData = preProcessor.preprocessData(self, rawData)

        # Step 3: Engineer features
        featuresDF = engineFeatures.engineFeatures(self, processedData)

        # Print feature summary and save
        print("\nEngineered Features Summary:")
        print(f"Total features: {len(featuresDF.columns)}")
        print(f"Total data points: {len(featuresDF)}")
        print("Feature names:", ", ".join(featuresDF.columns.tolist()[:5]) + "...")
        csv_dir = "Services/Models/data/data_exports"
        if not os.path.exists(csv_dir): os.makedirs(csv_dir)
        features_csv = f"{csv_dir}/raw_features.csv"
        featuresDF.to_csv(features_csv)
        print(f"Raw features saved to {features_csv}")

        # --- Step 4: Train LSTM Model ---
        print("\n--- Starting LSTM Model Training ---")
        try:
            # Train only the LSTM model
            lstm_training_results = simplePredictionModel.train_model(
                featuresDF,
                target_column='close',
                forecast_days=5,
                lstm_seq_length=10 # Can be adjusted
            )
        except Exception as e:
            print(f"FATAL ERROR during model training: {e}")
            import traceback
            traceback.print_exc()
            return # Stop execution if training fails fundamentally

        if lstm_training_results is None:
             print("LSTM training failed or was skipped. Exiting.")
             return

        # Extract results for the LSTM model
        lstm_model = lstm_training_results['model']
        model_scaler = lstm_training_results['scaler'] # Dictionary with 'feature' and 'target' scalers
        model_features = lstm_training_results['features']
        seq_length = lstm_training_results['seq_length']

        # Print evaluation metrics for the LSTM model
        print(f"\n--- LSTM Model Evaluation ---")
        print("  Test Metrics:")
        for metric, value in lstm_training_results['metrics'].items():
            print(f"    {metric.upper()}: {value:.4f}")
        if lstm_training_results.get('train_metrics'):
            print("  Train Metrics (Scaled):")
            for metric, value in lstm_training_results['train_metrics'].items():
                print(f"    {metric.upper()}: {value:.4f}")
        if lstm_training_results.get('validation_metrics'):
            print("  Validation Metrics (Scaled):")
            for metric, value in lstm_training_results['validation_metrics'].items():
                print(f"    {metric.upper()}: {value:.4f}")

        # --- Step 5: Backtesting (LSTM) ---
        print("\n--- Running Backtest ---")
        try:
             # Pass the feature scaler and seq_length
             backtest_results = simplePredictionModel.backtest_model(
                 lstm_model,
                 model_scaler['feature'], # Pass feature scaler
                 model_features,
                 featuresDF, # Pass original featuresDF
                 seq_length=seq_length # Pass seq_length
             )
        except Exception as e:
             print(f"Error during backtesting: {e}")
             import traceback
             traceback.print_exc()
             backtest_results = None

        # --- Step 6: Future Predictions (LSTM) ---
        print("\n--- Making Future Predictions ---")
        try:
            # Prepare input dictionary for LSTM prediction function
            lstm_pred_input = {
                'model': lstm_model,
                'feature_scaler': model_scaler['feature'],
                'target_scaler': model_scaler['target'],
                'seq_length': seq_length
            }
            # Ensure feature scaler has necessary attributes
            if not hasattr(lstm_pred_input['feature_scaler'], 'feature_names_in_') and hasattr(lstm_pred_input['feature_scaler'], 'feature_names'):
                 lstm_pred_input['feature_scaler'].feature_names_in_ = np.array(lstm_pred_input['feature_scaler'].feature_names, dtype=object)
                 lstm_pred_input['feature_scaler'].n_features_in_ = len(lstm_pred_input['feature_scaler'].feature_names)

            future_predictions = simplePredictionModel.lstmModel.predict_with_lstm(
                lstm_pred_input,
                featuresDF, # Pass original featuresDF
                days=5
            )
        except Exception as e:
            print(f"Error during future prediction: {e}")
            import traceback
            traceback.print_exc()
            future_predictions = pd.DataFrame()

        # Print detailed prediction info
        if not future_predictions.empty:
            print("\nDetailed Price Predictions:")
            if future_predictions.index[0] == future_predictions.index[0]:
                last_hist_price = featuresDF['close'].iloc[-1]
                for date, row in future_predictions.iterrows():
                    current_price = row['predicted_price']
                    if date == future_predictions.index[0]:
                        day_change_pct = ((current_price - last_hist_price) / last_hist_price) * 100
                        print(f"  {date.strftime('%Y-%m-%d')}: ${current_price:.2f} ({day_change_pct:+.2f}% from last known price)")
                    else:
                        idx = future_predictions.index.get_loc(date) - 1
                        prev_price = future_predictions['predicted_price'].iloc[idx]
                        day_change_pct = ((current_price - prev_price) / prev_price) * 100
                        print(f"  {date.strftime('%Y-%m-%d')}: ${current_price:.2f} ({day_change_pct:+.2f}% daily change)")

                first_pred = future_predictions['predicted_price'].iloc[0]
                last_pred = future_predictions['predicted_price'].iloc[-1]
                total_change_pct = ((last_pred - first_pred) / first_pred) * 100
                print(f"\nOverall {len(future_predictions)}-day prediction trend: {total_change_pct:+.2f}%")
        else:
            print("\nFuture predictions could not be generated.")

        # --- Step 7: Remove Feature Importance & Optimization ---
        print("\nSkipping feature importance and optimization (LSTM only).")
        feature_importance = None # Set to None as it's not calculated

        # --- Step 8: Visualization and Saving ---
        print("\n--- Final Visualization and Saving ---")
        if not future_predictions.empty:
            print("Creating visualization...")
            if 'avg_sentiment_score' not in featuresDF.columns: featuresDF['avg_sentiment_score'] = 0
            if 'fear_greed_index' not in featuresDF.columns: featuresDF['fear_greed_index'] = 50

            simplePredictionModel.visualize_predictions(
                featuresDF,
                future_predictions,
                save_path="Services/Models/data/crypto_price_prediction.png"
            )

            predictions_csv = f"{csv_dir}/price_predictions.csv"
            future_predictions.to_csv(predictions_csv)
            print(f"Predictions saved to {predictions_csv}")
        else:
            print("Skipping visualization and prediction saving due to errors.")

        # --- Step 9: Return Results ---
        results_dict = {
            "best_model_name": "lstm", # Hardcoded as LSTM is the only model
            "best_model_type": "lstm",
            "features_df": featuresDF,
            "training_results": lstm_training_results, # Contains LSTM results
            "predictions": future_predictions,
            "feature_importance": {} # Empty dict as it's not calculated
        }

        print("\nExecution finished.")
        return results_dict

if __name__ == "__main__":
    predictionModel = PredictionModel()
    results = predictionModel.runEverything()

