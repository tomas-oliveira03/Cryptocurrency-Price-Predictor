from datetime import datetime
from bson import CodecOptions
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
import requests
import os
from typing import Dict, List, Optional
import time
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cryptoCoinsInfo import getTopCoins

class Forum:
    
    def __init__(self, SHOW_LOGS=True):
        load_dotenv()
        
        self.SHOW_LOGS=SHOW_LOGS
        self.cryptoPanicAPIKey = os.getenv("CRYPTOPANIC_API_KEY")
        if not self.cryptoPanicAPIKey:
            raise ValueError("Please set cryptoPanicAPIKey environment variable first.")
        
        mongoDBURI = os.getenv("MONGODB_URI")
        if not mongoDBURI:
            raise ValueError("Please set the mongoDB environment variable first.")

        # Database connection
        mongoClient = MongoClient(mongoDBURI)
        self.mongoCollection = mongoClient['ASM'].get_collection('forum', codec_options=CodecOptions(tz_aware=True))

        self.headers = {
            'User-Agent': 'Python CryptoPanic API Client',
            'Accept': 'application/json'
        }
    
    
    def getAllInformation(self):
        topCoins = getTopCoins()
        listOfCryptoCurrencies = [None] + topCoins
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
                        
                        dataProcessed = self.processData(cryptoNews)
                        self.saveToMongo(dataProcessed)
                        if self.SHOW_LOGS: print(f"Latest news fetched for Crypto:{crypto}, Filter:{filter}, Kind:{kind}")
                        
                    except Exception as e:
                        if self.SHOW_LOGS: print(f"Exception: {e}")
                    
                    time.sleep(0.3)
            
        if self.SHOW_LOGS: print("Finished fetching all information.")
    
    
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
            response = requests.get("https://cryptopanic.com/api/v1/posts/", params=params, headers=self.headers)
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
        
        
    def processData(self, data: Dict) -> str:
        # Process the data before saving
        if 'results' in data and isinstance(data['results'], list):
            for result in data['results']:
                # Remove unnecessary entries
                if 'kind' in result:
                    del result['kind']
                if 'domain' in result:
                    del result['domain']
                if 'published_at' in result:
                    del result['published_at']
                if 'slug' in result:
                    del result['slug']

                # Simplify currencies to only include code and name
                if 'currencies' in result and isinstance(result['currencies'], list):
                    simplifiedCurrencies = []
                    for currency in result['currencies']:
                        if isinstance(currency, dict) and 'code' in currency and 'title' in currency:
                            simplifiedCurrencies.append({
                                'code': currency['code'],
                                'name': currency['title']
                            })
                    result['currencies'] = simplifiedCurrencies

                newResult = {}

                if 'id' in result:
                    newResult['id'] = result['id']

                if 'created_at' in result:
                    newResult['created_at'] = datetime.fromisoformat(result['created_at'].replace("Z", "+00:00"))
                
                if 'currencies' in result:
                    newResult['currencies'] = result['currencies']
                
                if 'source' in result and isinstance(result['source'], dict) and 'title' in result['source']:
                    newResult['source'] = result['source']['title'] 
                
                if 'title' in result:
                    newResult['title'] = result['title']
                
                if 'url' in result:
                    newResult['url'] = result['url']
                
                for key, value in result.items():
                    if key not in ['id', 'created_at', 'currencies', 'source', 'title', 'url']:
                        newResult[key] = value

                newResult["method"] = "Scrape"

                result.clear()
                result.update(newResult)
        
        return data["results"]


if __name__ == "__main__":
    forum = Forum()
    forum.getAllInformation()
