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
                        
                        # self.saveToMongo(cryptoNews["results"])
                        self.save_to_json(cryptoNews, "aaaa.json")
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
    
    
    # def saveToMongo(self, data):
    #     # Use upserts to avoid duplicates
    #     operations = [
    #         UpdateOne({"id": item["id"]}, {"$set": item}, upsert=True)
    #         for item in data
    #     ]
    #     numberOfInsertions = 0
        
    #     if operations:
    #         result = self.mongoCollection.bulk_write(operations)
    #         numberOfInsertions = result.upserted_count        
            
    #     return numberOfInsertions
        
    def save_to_json(self, data: Dict, filename: Optional[str] = None) -> str:
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'cryptopanic_data_{timestamp}.json'
        
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        
        # Process the data before saving
        if 'results' in data and isinstance(data['results'], list):
            for result in data['results']:
                # 1. Remove unnecessary entries
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
                    simplified_currencies = []
                    for currency in result['currencies']:
                        if isinstance(currency, dict) and 'code' in currency and 'title' in currency:
                            simplified_currencies.append({
                                'code': currency['code'],
                                'name': currency['title']
                            })
                    result['currencies'] = simplified_currencies

                # 2. Create a new ordered dictionary for the fields we want
                new_result = {}

                # Add id first if it exists
                if 'id' in result:
                    new_result['id'] = result['id']

                # Add created_at second if it exists
                if 'created_at' in result:
                    new_result['created_at'] = result['created_at']
                
                # Add currencies first
                if 'currencies' in result:
                    new_result['currencies'] = result['currencies']
                
                # Add source right after currencies
                if 'source' in result and isinstance(result['source'], dict) and 'title' in result['source']:
                    new_result['source'] = result['source']['title'] 
                
                # Add title next
                if 'title' in result:
                    new_result['title'] = result['title']
                
                # Add url next
                if 'url' in result:
                    new_result['url'] = result['url']
                
                # Add any other fields, ensuring that we avoid duplicating important fields
                for key, value in result.items():
                    if key not in ['id', 'created_at', 'currencies', 'source', 'title', 'url']:
                        new_result[key] = value

                new_result["method"] = "Scrape"
                # Replace the original result with our reordered one
                result.clear()
                result.update(new_result)
        
        with open(filepath, 'a', encoding='utf-8') as f:
            json.dump(data["results"], f, ensure_ascii=False, indent=4)
        
        print(f"Data saved to {filepath}")
        return filepath



if __name__ == "__main__":

    # Initialize the API client with your API key
    forum = Forum()
    forum.getAllInformation()
