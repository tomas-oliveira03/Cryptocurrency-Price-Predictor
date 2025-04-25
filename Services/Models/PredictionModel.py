import os
import sys
import pandas as pd
import numpy as np
from bson import CodecOptions
from pymongo import MongoClient
import json
import random
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Models.fetchData import getDataForPrediction
from Models.preProcessor import preprocessData
from Models.engineFeatures import engineFeatures
from Models.lstmModel import trainLstmModel, predictWithLstm
from Models.benchmark import coinBenchmark

class PredictionModel:
    def __init__(self, SHOW_LOGS=True):
        self.SHOW_LOGS=SHOW_LOGS
        mongoDBURI = os.getenv("MONGODB_URI")
        if not mongoDBURI:
            raise ValueError("Please set the MONGODB_URI environment variable first.")

        # Database connection
        mongoClient = MongoClient(mongoDBURI)
        
        self.cryptoFearGreedDB = mongoClient['ASM'].get_collection('crypto-fear-greed', codec_options=CodecOptions(tz_aware=True))
        self.detailedCryptoData = mongoClient['ASM'].get_collection('detailed-crypto-data', codec_options=CodecOptions(tz_aware=True))
        self.cryptoPriceDB = mongoClient['ASM'].get_collection('crypto-price', codec_options=CodecOptions(tz_aware=True))
        
        self.redditDB = mongoClient['ASM'].get_collection('reddit', codec_options=CodecOptions(tz_aware=True))
        self.forumDB = mongoClient['ASM'].get_collection('forum', codec_options=CodecOptions(tz_aware=True))
        self.articlesDB = mongoClient['ASM'].get_collection('articles', codec_options=CodecOptions(tz_aware=True))
        
        self.predictionsDB = mongoClient['ASM'].get_collection('predictions', codec_options=CodecOptions(tz_aware=True))
        
            
        

    def runEverything(self, cryptoCoin, forcastDays, initialFetchDays):
        self.setSeeding(42)
        
        # Save raw feature data first
        dataDir = "Services/Models/data"
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)


        # Step 1: Get raw data for prediction
        rawData = getDataForPrediction(self, cryptoCoin=cryptoCoin, numberOfPastDaysOfData=initialFetchDays)
        
        # Step 2: Preprocess the data
        processedData = preprocessData(self, rawData)
        
        # Step 3: Engineer features
        featuresDF = engineFeatures(self, processedData)
        
        # Step 4: Train LSTM model (Initial training)
        print("\n--- Training Initial LSTM Model ---")
        
        # Use a smaller sequence length if dataset is smaller
        seq_length = min(10, len(featuresDF) // 5)  # Adjust sequence length based on data size
        if seq_length < 1: seq_length = 1 # Ensure seq_length is at least 1

        lstm_results = trainLstmModel(
            featuresDF,
            target_column='close',
            forecast_days=1, # Typically train LSTM for 1-step ahead prediction
            test_size=0.2,   # Use a portion for validation during training
            seq_length=seq_length
        )            
            
        # Step 5: Retrain LSTM on full dataset for final model
        print("\n--- Retraining LSTM Model on Full Historical Data ---")
        
        # Retrain LSTM on the entire dataset
        retrain_seq_length = min(10, len(featuresDF) // 10)
        if retrain_seq_length < 1: retrain_seq_length = 1

        final_lstm_model = trainLstmModel(
            featuresDF, # Use full dataset
            target_column='close',
            forecast_days=1,  # Still 1-step ahead prediction
            test_size=0.01,   # Minimal test set for final training
            seq_length=retrain_seq_length
        )
        print("LSTM retrained successfully on full data.")

        # Step 6: Predict Future with LSTM
        print(f"\n--- Predicting Next {forcastDays} Days using LSTM Model ---")

        future_predictions = predictWithLstm(
            final_lstm_model,
            featuresDF,
            days=forcastDays
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
            print(f"\nOverall {forcastDays}-day prediction trend: {total_change_pct:+.2f}%")
        else:
            print("  No future predictions were generated.")

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
        
        # Extract model benchmarking results
        model_metrics = lstm_results['metrics']
        
        coinBenchmarkData = coinBenchmark(cryptoCoin, self.predictionsDB, self.cryptoPriceDB)
        
        json_data = {
            "coin": cryptoCoin,
            "historical_price": historical_price_data,
            "predicted_price": predicted_price_data,
            "positive_sentiment_ratio": positive_sentiment_data,
            "model_benchmarks": {
                "mse": model_metrics.get('mse', 0),
                "rmse": model_metrics.get('rmse', 0),
                "mae": model_metrics.get('mae', 0),
                "r2": model_metrics.get('r2', 0),
                "mape": model_metrics.get('mape', 0)
            },
            "prediction_benchmarks": coinBenchmarkData
        }
        
        self.saveToMongo(json_data)
        
        # Return results dictionary
        results_dict = {
            "features": featuresDF,
            "lstm_results": lstm_results,
            "predictions": future_predictions,
            "json_data": json_data,  # Add the actual JSON data to the results
            "forcastDays": forcastDays
        }
            
        return results_dict


    def saveToMongo(self, data):
        try:
            # Convert all date fields in the data to datetime objects with timezone info
            def convert_dates(obj):
                if isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, dict) and 'date' in item:
                            item['date'] = pd.to_datetime(item['date']).tz_localize('UTC')
                return obj

            # Apply the conversion to specific fields
            data['historical_price'] = convert_dates(data.get('historical_price', []))
            data['predicted_price'] = convert_dates(data.get('predicted_price', []))
            data['positive_sentiment_ratio'] = convert_dates(data.get('positive_sentiment_ratio', []))

            # Add a timestamp to the data for sorting purposes
            data['date'] = pd.Timestamp.now(tz='UTC')

            # Insert the content into the predictionsDB collection
            self.predictionsDB.insert_one(data)
            print(f"Content successfully saved to MongoDB with timestamp {data['date']}")
        except Exception as e:
            print(f"Failed to save content to MongoDB: {e}")
            raise


    def setSeeding(self, seed=42):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        print(f"Seeds set to {seed} for random, numpy, and tensorflow (if available).")


if __name__ == "__main__":
    try:

        
        # --- Configuration ---
        forcastDays = 7   # Days to predict into the future
        initialFetchDays = 365 * 2 # Fetch ample history initially (e.g., 2 years)
        # -------------------
        
        # Get the list of top coins
        coins = ['BTC', 'ETH', 'XRP', 'BNB', 'SOL', 'DOGE', 'TRX', 'ADA']
        
        for cryptoCoin in coins:
            print(f"\n\n{'='*50}")
            print(f"Running prediction model for {cryptoCoin}")
            print(f"{'='*50}\n")
            
            predictionModel = PredictionModel()
            results = predictionModel.runEverything(cryptoCoin, forcastDays, initialFetchDays)
            
            print("\n--- Final Results Summary for", cryptoCoin, "---")
            
            # Access config values from the results dictionary
            fc_days = results.get('forcastDays', 'N/A')
    
            print(f"Model used: LSTM")
            print("LSTM Model Metrics:")
            lstm_metrics = results.get('lstm_results', {}).get('metrics', {})
            for metric, value in lstm_metrics.items():
                print(f"  {metric.upper()}: {value:.4f}")
                
            # Add metrics evaluation
            r2 = lstm_metrics.get('r2', 0)
            mape = lstm_metrics.get('mape', 0)
            print("\nMetrics Evaluation:")
            if r2 > 0.5:
                print(f"  R² of {r2:.4f} shows the model has moderate predictive power")
            else:
                print(f"  R² of {r2:.4f} suggests the model has limited predictive power")
                
            if mape < 5:
                print(f"  MAPE of {mape:.4f}% indicates relatively accurate predictions")
            elif mape < 10:
                print(f"  MAPE of {mape:.4f}% indicates acceptable prediction accuracy")
            else:
                print(f"  MAPE of {mape:.4f}% indicates high prediction errors")
    
            print(f"Final {fc_days}-day Predictions:")
            print(results.get('predictions', pd.DataFrame()))
            
            # Print confirmation that JSON contains benchmarks
            print(f"\nJSON export includes full model benchmarking metrics")
            
    except ValueError as ve:
        print(f"\n--- Execution Failed (ValueError) ---")
        print(ve)
    except Exception as e:
        print(f"\n--- Execution Failed (Unexpected Error) ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()