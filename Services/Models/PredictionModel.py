import os
from datetime import datetime, timedelta
import pandas as pd

from bson import CodecOptions
from pymongo import MongoClient
import fetchData


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
        

    def getDataForPrediction(self, cryptoCoin):       

        startDate = datetime.now() - timedelta(days=30)
        endDate = datetime.now() - timedelta(days=5)
        
        # Get data from all sources
        cryptoData = fetchData.getCryptoPriceData(self, startDate=startDate, cryptoSymbol=cryptoCoin)
        fearGreedData = fetchData.getFearGreedData(self, startDate=startDate)
        
        # Get social media data related to this cryptocurrency
        redditData = fetchData.getRedditData(self, startDate=startDate, cryptoSymbol=cryptoCoin)
        forumData = fetchData.getForumData(self, startDate=startDate, cryptoSymbol=cryptoCoin)
        articlesData = fetchData.getArticlesData(self, startDate=startDate, cryptoSymbol=cryptoCoin)

        datasets = {
            "price_data": cryptoData,
            "fear_greed_data": fearGreedData,
            "reddit_data": redditData,
            "forum_data": forumData,
            "articles": articlesData
        }

        return datasets
    
    def preprocessData(self, rawData):
        """
        Convert raw MongoDB data to pandas DataFrames and preprocess it
        
        Args:
            rawData: Dictionary containing the raw data from MongoDB
            
        Returns:
            Dictionary containing preprocessed pandas DataFrames
        """
        processedData = {}
        
        # Process price data
        if rawData["price_data"]:
            priceDF = pd.DataFrame(rawData["price_data"])
            priceDF['date'] = pd.to_datetime(priceDF['date'])
            priceDF.set_index('date', inplace=True)
            priceDF.sort_index(inplace=True)
            processedData["price_df"] = priceDF
        else:
            processedData["price_df"] = pd.DataFrame()
            
        # Process fear and greed data
        if rawData["fear_greed_data"]:
            fearGreedDF = pd.DataFrame(rawData["fear_greed_data"])
            fearGreedDF['date'] = pd.to_datetime(fearGreedDF['date'])
            fearGreedDF.set_index('date', inplace=True)
            fearGreedDF.sort_index(inplace=True)
            processedData["fear_greed_df"] = fearGreedDF
        else:
            processedData["fear_greed_df"] = pd.DataFrame()
            
        # Process Reddit data
        if rawData["reddit_data"]:
            redditDF = pd.DataFrame(rawData["reddit_data"])
            redditDF['created_at'] = pd.to_datetime(redditDF['created_at'])
            
            # Extract just the sentiment label if available
            if 'sentiment' in redditDF.columns:
                try:
                    # Extract sentiment label directly
                    redditDF['sentiment_category'] = redditDF['sentiment'].apply(
                        lambda x: x.get('label', 'Neutral') if isinstance(x, dict) else 'Neutral'
                    )
                except Exception as e:
                    print(f"Error extracting Reddit sentiment: {e}")
            
            processedData["reddit_df"] = redditDF
        else:
            processedData["reddit_df"] = pd.DataFrame()
            
        # Process Forum data
        if rawData["forum_data"]:
            forumDF = pd.DataFrame(rawData["forum_data"])
            forumDF['created_at'] = pd.to_datetime(forumDF['created_at'])
            
            # Extract just the sentiment label if available
            if 'sentiment' in forumDF.columns:
                try:
                    # Extract sentiment label directly
                    forumDF['sentiment_category'] = forumDF['sentiment'].apply(
                        lambda x: x.get('label', 'Neutral') if isinstance(x, dict) else 'Neutral'
                    )
                except Exception as e:
                    print(f"Error extracting Forum sentiment: {e}")
            
            processedData["forum_df"] = forumDF
        else:
            processedData["forum_df"] = pd.DataFrame()
            
        return processedData


if __name__ == "__main__":
    # Example usage
    predictionModel = PredictionModel()
    
    # Get raw data for prediction
    print("Fetching data...")
    rawData = predictionModel.getDataForPrediction(cryptoCoin="BTC")
    
    # # Preprocess the data
    # print("Preprocessing data...")
    # processedData = predictionModel.preprocessData(rawData)
    
    # # Save each DataFrame to a CSV file
    # print("\nSaving data to CSV files...")
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # csv_dir = "data_exports"
    
    # # Create the directory if it doesn't exist
    # if not os.path.exists(csv_dir):
    #     os.makedirs(csv_dir)
    
    # # Export each DataFrame to a CSV file
    # for name, df in processedData.items():
    #     if not df.empty:
    #         csv_filename = f"{csv_dir}/{name}_{timestamp}.csv"
    #         df.to_csv(csv_filename)
    #         print(f"  Saved {name} to {csv_filename}")
    #     else:
    #         print(f"  {name} is empty, no CSV file created")
    
    # # Print table with first 5 rows of each dataset
    # print("\n" + "="*80)
    # print("FIRST 5 ROWS OF EACH DATASET")
    # print("="*80)
    
    # for name, df in processedData.items():
    #     if not df.empty:
    #         print(f"\n{name.upper()} - First 5 rows:")
    #         print("-"*80)
    #         print(df.head(5))
    #         print("-"*80)
    #     else:
    #         print(f"\n{name.upper()}: Empty DataFrame")
    
    # # Print general information about the processed data
    # print("\nProcessed Data Information:")
    # for name, df in processedData.items():
    #     if not df.empty:
    #         print(f"\n{name}:")
    #         print(f"  Shape: {df.shape}")
    #         print(f"  Columns: {df.columns.tolist()}")
    #     else:
    #         print(f"\n{name}: Empty DataFrame")
    
    # print("\nPreprocessing complete!")