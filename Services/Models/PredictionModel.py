import os
from datetime import datetime, timedelta

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
        

    
    
    
    
    
    
    
    
    def getDataForPrediction(self):
        """
        Get all relevant data needed for price prediction for a specific cryptocurrency
        
        Args:
            cryptoSymbol: Symbol of the cryptocurrency (e.g., "BTC", "ETH")
            daysAgo: Number of days of historical data to retrieve
            
        Returns:
            Dictionary containing all data needed for prediction
        """
        # Calculate start date
        
        cryptoSymbol="BTC"
        cryptoSymbolReddit="Bitcoin"
        
        startDate = datetime.now() - timedelta(days=30)
        endDate = datetime.now() - timedelta(days=5)
        
        # Get data from all sources
        cryptoData = fetchData.getCryptoPriceData(self, cryptoSymbol=cryptoSymbol, startDate=startDate)
        fearGreedData = fetchData.getFearGreedData(self, startDate=startDate)
        
        # Get social media data related to this cryptocurrency
        redditData = fetchData.getRedditData(self, startDate=startDate, subreddit=cryptoSymbolReddit)
        forumData = fetchData.getForumData(self, cryptoSymbol=cryptoSymbol, startDate=startDate)



        # Return all data in a dictionary
        return {
            "price_data": cryptoData,
            "fear_greed_data": fearGreedData,
            "reddit_data":  redditData,
            "forum_data":  forumData
        }


if __name__ == "__main__":
    # Example usage
    predictionModel = PredictionModel()
    
    # Get data for prediction
    data = predictionModel.getDataForPrediction()
    
    # Print the retrieved data
    print(data)