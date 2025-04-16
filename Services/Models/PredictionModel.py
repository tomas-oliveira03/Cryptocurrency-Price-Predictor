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
        # Step 1: Get raw data for prediction
        rawData = fetchData.getDataForPrediction(self, cryptoCoin="BTC", numberOfPastDaysOfData=100)
        
        # Print the number of entries in each key of rawData
        print("Raw data summary:")
        total_price_data = min(len(rawData.get("price_data", [])), len(rawData.get("fear_greed_data", [])))
        data_count = sum(len(rawData[key]) for key in ["reddit_data", "forum_data", "articles_data"] if key in rawData and rawData[key] is not None)
        
        for key, value in rawData.items():
            if isinstance(value, pd.DataFrame):
                print(f"  {key}: {len(value)} entries")
            else:
                print(f"  {key}: {len(value) if hasattr(value, '__len__') else 'N/A'} entries")
        
        print(f"\nTotal price data (min of price_data and fear_greed_data): {total_price_data}")
        print(f"Data count (sum of reddit, forum, and articles): {data_count}\n")
        
        
        # Step 2: Preprocess the data
        processedData = preProcessor.preprocessData(self, rawData)
        
        # Step 3: Engineer features
        featuresDF = engineFeatures.engineFeatures(self, processedData)
        
        # Step 4: Train prediction model and evaluate
        model_results = simplePredictionModel.train_model(
            featuresDF, 
            target_column='close', 
            forecast_days=5
        )
        
        # Print model evaluation metrics
        print("\nModel Evaluation Metrics:")
        for metric, value in model_results['metrics'].items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        # Step 5: Make future predictions
        print("\nMaking predictions for the next 5 days...")
        future_predictions = simplePredictionModel.make_future_predictions(
            model_results['model'],
            model_results['scaler'],
            model_results['features'],
            featuresDF,
            days=5
        )
        
        # Print the predictions
        print("\nPrice Predictions:")
        for date, row in future_predictions.iterrows():
            print(f"  {date.strftime('%Y-%m-%d')}: ${row['predicted_price']:.2f}")
        
        # Step 6: Visualize results
        print("\nCreating visualization...")
        simplePredictionModel.visualize_predictions(
            featuresDF[['close']],
            future_predictions,
            save_path="crypto_price_prediction.png"
        )
        
        # Save the engineered features to CSV
        csv_dir = "data_exports"
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        

        # Save predictions to CSV
        predictions_csv = f"{csv_dir}/price_predictions.csv"
        future_predictions.to_csv(predictions_csv)
        print(f"Predictions saved to {predictions_csv}")
        
        return {
            "features": featuresDF,
            "model_results": model_results,
            "predictions": future_predictions
        }

if __name__ == "__main__":
    # Example usage
    predictionModel = PredictionModel()
    results = predictionModel.runEverything()

