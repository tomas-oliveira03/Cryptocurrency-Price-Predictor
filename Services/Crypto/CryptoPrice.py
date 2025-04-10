import os
from bson import CodecOptions
from pymongo import MongoClient, UpdateOne
import pytz
import requests
from dotenv import load_dotenv
from datetime import datetime

class CryptoPrice:
    def __init__(self, SHOW_LOGS=True):
        self.SHOW_LOGS=SHOW_LOGS
        
        load_dotenv()
        self.coinDeskAPIKey = os.getenv("COINDESK_API_KEY")
        self.coinMarketCapAPIKey = os.getenv('COINMARKETCAP_API_KEY')
        self.currencySymbol = "USD" 
    
        if(not self.coinDeskAPIKey or not self.coinMarketCapAPIKey):
            raise ValueError("Please set the COINDESK_KEY and COINMARKETCAP_KEY environment variables first.")
        
        mongoDBURI = os.getenv("MONGODB_URI")
        if not mongoDBURI:
            raise ValueError("Please set the MONGODB_URI environment variable first.")

        # MongoDB connection
        mongoClient = MongoClient(mongoDBURI)
        self.mongoCollection = mongoClient['ASM'].get_collection('crypto-price', codec_options=CodecOptions(tz_aware=True))


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
        
    
    def getTopCoins(self, topK=10):
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
        headers = {
            'X-CMC_PRO_API_KEY': self.coinMarketCapAPIKey, 
            'Accept': 'application/json'
        }

        params = {
            'limit': topK,                        # Top K coins
            'convert': self.currencySymbol,       # Convert prices to USD
            'sort': 'market_cap',                 # Sort by market cap
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            coins = response.json().get('data', [])
            topCoins = [coin['symbol'] for coin in coins]  # Get the symbol of the top 10 coins
            return topCoins
        else:
            if self.SHOW_LOGS: print(f"Error fetching CoinMarketCap data: {response.status_code}")
            return []


    def fetchCoinsData(self):
        # Get top 10 coins by market cap
        topCoins = self.getTopCoins()
        if not topCoins:
            raise ValueError("Failed to fetch top coins.")
        
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
        
        


# Example usage
if __name__ == "__main__":
    # Create an instance of CryptoPrice with the API key
    cryptoApi = CryptoPrice()

    # Fetch and save historical data to MongoDB
    cryptoApi.fetchCoinsData()
