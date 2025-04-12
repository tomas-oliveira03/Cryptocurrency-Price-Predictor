from bson import CodecOptions
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
import requests
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import time

class Forum:
    
    def __init__(self):
        load_dotenv()
        self.cryptoPanicAPIKey = os.getenv("CRYPTOPANIC_API_KEY")
        if not self.cryptoPanicAPIKey:
            raise ValueError("Please set cryptoPanicAPIKey environment variable first.")
        
        mongoDBURI = os.getenv("MONGODB_URI")
        if not mongoDBURI:
            raise ValueError("Please set the mongoDB environment variable first.")

        # Database connection
        mongoClient = MongoClient(mongoDBURI)
        self.mongoCollection = mongoClient['ASM'].get_collection('forum', codec_options=CodecOptions(tz_aware=True))

        self.base_url = "https://cryptopanic.com/api/v1"
        self.headers = {
            'User-Agent': 'Python CryptoPanic API Client',
            'Accept': 'application/json'
        }
    
    def getAllInformation(self):
        
        listOfCryptoCurrencies = [None, 'BTC', 'ETH', 'USDT', 'XRP', 'BNB', 'SOL', 'USDC', 'DOGE', 'TRX', 'ADA']
        listOfFilters = ['rising', 'hot', 'bullish', 'bearish', 'important', 'saved', 'lol']
        listOfKindOfInfo = ['news', 'media']
            
        for crypto in listOfCryptoCurrencies:
            for filter in listOfFilters:
                for kind in listOfKindOfInfo:
                    
                    try:
                        if crypto is None:
                            cryptoNews = self.getPostsByFilter(filter_type=filter, kind=kind)
                        else:
                            cryptoNews = self.getPostsByFilter(filter_type=filter, kind=kind, currencies=[crypto])
                        
                        self.saveToMongo(cryptoNews["results"])
                        print("Latest news", len(cryptoNews["results"]))
                        
                    except Exception as e:
                        print(f"Exception: {e}")
                    
                    time.sleep(0.3)
            
        print("Finished fetching all information.")
    
    
    def getPostsByFilter(self, 
                 currencies: Optional[List[str]] = None, # BTC|ETH|SOL
                 filter_type: Optional[str] = None, # rising|hot|bullish|bearish|important|saved|lol
                 kind: Optional[str] = None, # news|media
                 limit: int = 50) -> Dict:
        
        params = {"auth_token": self.cryptoPanicAPIKey, "limit": limit}
            
        if currencies:
            params["currencies"] = ",".join(currencies)
            
        if filter_type:
            params["filter"] = filter_type
            
        if kind:
            params["kind"] = kind

        
        try:
            response = requests.get(f"{self.base_url}/posts/", params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise
    
    
    def saveToMongo(self, data):
        # Use upserts to avoid duplicates
        operations = [
            UpdateOne({"id": item["id"]}, {"$set": item}, upsert=True)
            for item in data
        ]
        numberOfInsertions = 0
        
        if operations:
            result = self.mongoCollection.bulk_write(operations)
            numberOfInsertions = result.upserted_count        
            
        return numberOfInsertions
       


if __name__ == "__main__":

    # Initialize the API client with your API key
    forum = Forum()
    forum.getAllInformation()
