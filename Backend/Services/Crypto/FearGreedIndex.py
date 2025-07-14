from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
import pytz
import requests
from datetime import datetime
from bson.codec_options import CodecOptions
import os

class FearGreedIndex:
    def __init__(self, SHOW_LOGS=True):
        self.SHOW_LOGS=SHOW_LOGS
        mongoDBURI = os.getenv("MONGODB_URI")
        if not mongoDBURI:
            raise ValueError("Please set the MONGODB_URI environment variable first.")

        # Database connection
        mongoClient = MongoClient(mongoDBURI)
        self.mongoCollection = mongoClient['ASM'].get_collection('crypto-fear-greed', codec_options=CodecOptions(tz_aware=True))


    def fetchFearGreedIndexData(self):
        response = requests.get("https://api.alternative.me/fng/?limit=365&format=json")

        if response.status_code == 200:
            rawData = response.json()["data"]

            formattedData = []
            for item in rawData:
                utcTime = datetime.fromtimestamp(int(item["timestamp"])).replace(tzinfo=pytz.UTC)
                formattedData.append({
                    "date": utcTime,
                    "value": int(item["value"]),
                    "classification": item["value_classification"]
                })

            databaseInfo = self.saveToMongo(formattedData)
            if self.SHOW_LOGS: print(f"FearGreed index data saved to MongoDB successfully.")
            return databaseInfo
        
        else:
            if self.SHOW_LOGS: print(f"Failed to fetch data: {response.status_code}")
            raise ValueError("Failed to fetch fear-greed index.")


    def saveToMongo(self, data):
        # Use upserts to avoid duplicates
        operations = [
            UpdateOne({"date": item["date"]}, {"$set": item}, upsert=True)
            for item in data
        ]
        numberOfInsertions = 0
        
        if operations:
            result = self.mongoCollection.bulk_write(operations)
            numberOfInsertions = result.upserted_count        
            
        return numberOfInsertions
       
