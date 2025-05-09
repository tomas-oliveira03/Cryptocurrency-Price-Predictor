import os
from bson import CodecOptions
from pymongo import MongoClient, UpdateOne
import pytz
import requests
from dotenv import load_dotenv
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cryptoCoinsInfo import getTopCoins

class DetailedCryptoData:
    def __init__(self, SHOW_LOGS=True):
        self.SHOW_LOGS=SHOW_LOGS
        self.coinDeskAPIKey = os.getenv("COINDESK_API_KEY")
        self.currencySymbol = "USD" 
    
        if(not self.coinDeskAPIKey):
            raise ValueError("Please set the COINDESK_KEY and COINMARKETCAP_KEY environment variables first.")
        
        mongoDBURI = os.getenv("MONGODB_URI")
        if not mongoDBURI:
            raise ValueError("Please set the MONGODB_URI environment variable first.")

        # MongoDB connection
        mongoClient = MongoClient(mongoDBURI)
        self.mongoCollection = mongoClient['ASM'].get_collection('detailed-crypto-data', codec_options=CodecOptions(tz_aware=True))


    def fetchHistoricalData(self, coinSymbol, limit=365):
        # Set the parameters for the API request
        params = {
            "fsym": coinSymbol,          # Cryptocurrency symbol (e.g., BTC)
            "tsym": self.currencySymbol, # Fiat currency symbol (e.g., USD)
            "limit": limit,              # Number of data points (days)
        }

        # Set the headers with the API key
        headers = {
            "Authorization": f"Apikey {self.coinDeskAPIKey}"
        }

        # Make the API request
        response = requests.get("https://min-api.cryptocompare.com/data/v2/histoday", headers=headers, params=params)

        if response.status_code == 200:
            data = response.json().get("Data", {}).get("Data", [])

            # Convert the UNIX timestamp to a readable date
            for dayData in data:
                utcTime = datetime.fromtimestamp(dayData["time"]).replace(tzinfo=pytz.UTC)
                dayData["date"] = utcTime

                # Add fields
                dayData["cryptoCurrency"] = coinSymbol
                dayData["currency"] = self.currencySymbol

                # Calculate average price
                high = dayData["high"]
                low = dayData["low"]
                close = dayData["close"]
                averagePrice = (high + low + close) / 3
                dayData["averagePrice"] = averagePrice

            if self.SHOW_LOGS: print(f"Fetched data for {coinSymbol}. Entries: {len(data)}")
            return data
        
        else:
            if self.SHOW_LOGS: print(f"Failed to fetch data: {response.status_code}")
            return None

    def fetchCoinsDataFullAnalysis(self):
        # Get top coins by market cap
        topCoins = getTopCoins()
        
        if self.SHOW_LOGS: print(f"Top 10 traded coins: {topCoins}")

        allHistoricalData = []

        for coinSymbol in topCoins:
            historicalData = self.fetchHistoricalData(coinSymbol, limit=365)

            if historicalData:
                allHistoricalData.extend(historicalData)

        # Save all fetched data to MongoDB
        databaseInfo = self.saveToMongo(allHistoricalData)
        if self.SHOW_LOGS: print("Crypto prices data saved to MongoDB successfully.")
        return databaseInfo
        
        
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
