import os
import sys
import pandas as pd
import numpy as np
from bson import CodecOptions
from pymongo import MongoClient
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')   

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models.fetchData import getDataForPrediction
from Models.preProcessor import preprocessData
from Models.engineFeatures import engineFeatures
from Models.lstmModel import trainLstmModel, predictWithLstm
from Models.benchmark import coinBenchmark

from utils.cryptoCoinsInfo import getTopCoins


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
        
    
    def runModelForEveryCrypto(self, forcastDays, initialFetchDays):
        topCoins = getTopCoins()
        allCoinsData = []
        for coin in topCoins:
            try:
                coinData = self.runModelForCrypto(coin, forcastDays, initialFetchDays)
                allCoinsData.append(coinData)
            except Exception as e:
                if self.SHOW_LOGS: print(f"Error processing {coin}: {e}")
                continue
            
        return allCoinsData 
        


    def runModelForCrypto(self, cryptoCoin, forcastDays, initialFetchDays):
        agentName=f"\033[38;5;88m[PredictionModel]\033[0m"
        if self.SHOW_LOGS: print(f"{agentName} Running prediction model for crypto coin: {cryptoCoin}")
        
        self.setSeeding(42)
        
        # Step 1: Get raw data for prediction
        rawData = getDataForPrediction(self, cryptoCoin=cryptoCoin, numberOfPastDaysOfData=initialFetchDays)
        
        # Step 2: Preprocess the data
        processedData = preprocessData(self, rawData, self.SHOW_LOGS)
        
        # Step 3: Engineer features
        featuresDF = engineFeatures(self, processedData)
        
        # Step 4: Train LSTM model (Initial training)
        if self.SHOW_LOGS: print("\n--- Training Initial LSTM Model ---")
        
        # Use a smaller sequence length if dataset is smaller
        seq_length = min(10, len(featuresDF) // 5)  # Adjust sequence length based on data size
        if seq_length < 1: seq_length = 1 # Ensure seq_length is at least 1

        lstm_results = trainLstmModel(
            featuresDF,
            target_column='close',
            forecast_days=1, # Typically train LSTM for 1-step ahead prediction
            test_size=0.2,   # Use a portion for validation during training
            seq_length=seq_length,
            SHOW_LOGS=self.SHOW_LOGS
        )            
            
        # Step 5: Retrain LSTM on full dataset for final model
        if self.SHOW_LOGS: print("\n--- Retraining LSTM Model on Full Historical Data ---")
        
        # Retrain LSTM on the entire dataset
        retrain_seq_length = min(10, len(featuresDF) // 10)
        if retrain_seq_length < 1: retrain_seq_length = 1

        final_lstm_model = trainLstmModel(
            featuresDF, # Use full dataset
            target_column='close',
            forecast_days=1,  # Still 1-step ahead prediction
            test_size=0.01,   # Minimal test set for final training
            seq_length=retrain_seq_length,
            SHOW_LOGS=self.SHOW_LOGS
        )
        if self.SHOW_LOGS: print("LSTM retrained successfully on full data.")

        # Step 6: Predict Future with LSTM
        if self.SHOW_LOGS: print(f"\n--- Predicting Next {forcastDays} Days using LSTM Model ---")

        future_predictions = predictWithLstm(
            final_lstm_model,
            featuresDF,
            days=forcastDays,
            SHOW_LOGS=self.SHOW_LOGS
        )

        # Step 7: Output Results
        # Print detailed prediction information
        if self.SHOW_LOGS: print("\nDetailed Future Price Predictions:")
        if not future_predictions.empty:
            for date, row in future_predictions.iterrows():
                if self.SHOW_LOGS: print(f"  {date.strftime('%Y-%m-%d')}: ${row['predicted_price']:.2f}")
            
            # Calculate overall prediction trend
            first_pred = future_predictions['predicted_price'].iloc[0]
            last_pred = future_predictions['predicted_price'].iloc[-1]
            total_change_pct = ((last_pred - first_pred) / first_pred) * 100 if first_pred != 0 else 0
            if self.SHOW_LOGS: print(f"\nOverall {forcastDays}-day prediction trend: {total_change_pct:+.2f}%")
        else:
            if self.SHOW_LOGS: print("  No future predictions were generated.")

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
        
        cryptoCoinPredictionForNextDay = predicted_price_data[0]
            
        result = {
            "coin": cryptoCoin,
            "price": cryptoCoinPredictionForNextDay["price"]
        }
        
        return result


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
            if self.SHOW_LOGS: print(f"Content successfully saved to MongoDB with timestamp {data['date']}")
        except Exception as e:
            if self.SHOW_LOGS: print(f"Failed to save content to MongoDB: {e}")
            raise


    def setSeeding(self, seed=42):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)


if __name__ == "__main__":

    forcastDays = 7   # Days to predict into the future
    initialFetchDays = 365 * 2 # Fetch ample history initially (e.g., 2 years)
    
    # Get the list of top coins
    predictionModel = PredictionModel()
    coinData = predictionModel.runModelForEveryCrypto(forcastDays, initialFetchDays)
    print(coinData)
