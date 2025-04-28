
from datetime import datetime, timezone
import os
import sys
from bson import CodecOptions
from dotenv import load_dotenv
from flask import json
from pymongo import MongoClient, UpdateOne
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cryptoCoinsInfo import getTickerToFullNameMap, getFullNameToTickerMap

class CryptoPrice:
    def __init__(self, SHOW_LOGS=True):
        load_dotenv()
        
        self.SHOW_LOGS=SHOW_LOGS
        self.currencySymbol = "usd" 
        
        mongoDBURI = os.getenv("MONGODB_URI")
        if not mongoDBURI:
            raise ValueError("Please set the MONGODB_URI environment variable first.")

        # MongoDB connection
        mongoClient = MongoClient(mongoDBURI)
        self.mongoCollection = mongoClient['ASM'].get_collection('crypto-price', codec_options=CodecOptions(tz_aware=True))


    def fetchTopCoinsPrices(self):
        ids = ','.join(getTickerToFullNameMap().values())
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies={self.currencySymbol}"
        response = requests.get(url)

        if response.status_code == 200:
            allCoinsData = response.json()
            
            tickerNameMap = getFullNameToTickerMap()
            finalResults = []
            for coin, data in allCoinsData.items():
                if coin in tickerNameMap:
                    tickerName = tickerNameMap[coin]
                    finalResults.append({
                        "cryptoCurrency": tickerName,
                        "price": data[self.currencySymbol],
                        "date": datetime.now(timezone.utc)
                    })
                    
            databaseInfo = self.saveToMongo(finalResults)
            if self.SHOW_LOGS: print("Crypto prices data saved to MongoDB successfully.")
            
            # Send Webhook to notify frontend
            url = "http://localhost:3001/api/crypto/update"
            
            modifiedResults = []

            for result in finalResults:
                modifiedResult = {
                    "coin": result["cryptoCurrency"],  
                    "price": result["price"]           
                }
                modifiedResults.append(modifiedResult)
            
            payload = modifiedResults
            rawPayload = json.dumps(payload)
            
            try:
                # Send the POST request with a timeout of 10 seconds
                response = requests.post(url, data=rawPayload)

                # Check if the response status code indicates success
                if response.status_code == 200:
                    if self.SHOW_LOGS:
                        print("Status:", response.status_code)
                        print("Response:", response.text)
                else:
                    if self.SHOW_LOGS:
                        print(f"Error: Received status code {response.status_code}")
                        print("Response:", response.text)

            except requests.exceptions.RequestException as e:
                if self.SHOW_LOGS:
                    print(f"Error: Request failed - {e}")
            return modifiedResults
            
        else:
            raise Exception(f"Failed to fetch data: {response.status_code}")


    def saveToMongo(self, data):
        # Use upserts to avoid duplicates
        operations = [
            UpdateOne(
                {"date": item["date"], "cryptoCurrency": item["cryptoCurrency"]}, 
                {"$set": item}, 
                upsert=True
            )
            for item in data
        ]
        numberOfInsertions = 0
        
        if operations:
            result = self.mongoCollection.bulk_write(operations)
            numberOfInsertions = result.upserted_count        
            
        return numberOfInsertions


if __name__ == "__main__":
    detailedCryptoData = CryptoPrice()
    data = detailedCryptoData.fetchTopCoinsPrices()