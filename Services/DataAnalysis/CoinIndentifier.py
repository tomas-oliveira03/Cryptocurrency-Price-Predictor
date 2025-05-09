import os
import re
import sys
from bson import CodecOptions
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cryptoCoinsInfo import getTickerToFullNameMap


class CoinIdentifier:
    def __init__(self, SHOW_LOGS=True):
        self.SHOW_LOGS=SHOW_LOGS
        mongoDBURI = os.getenv("MONGODB_URI")
        if not mongoDBURI:
            raise ValueError("Please set the MONGODB_URI environment variable first.")

        # Database connection
        mongoClient = MongoClient(mongoDBURI)
        
        self.allMongoCollectionsDict = {
            "reddit": mongoClient['ASM'].get_collection('reddit', codec_options=CodecOptions(tz_aware=True)),
            "articles": mongoClient['ASM'].get_collection('articles', codec_options=CodecOptions(tz_aware=True)),
        }
        
        self.allTopCoins = getTickerToFullNameMap()


    def idetifyCoinsForAllCollections(self, collectionName):
        collection = self.allMongoCollectionsDict.get(collectionName)
        if collection is None:
            if self.SHOW_LOGS: print(f"Collection: {collectionName}, does not need coin identifier, it already has it by default...")
            return
    
        if self.SHOW_LOGS: print("Coin identifier for collection:", collectionName)
        self.identifyCoinsForCollection(collection, collectionName)
        

    def identifyCoinsForCollection(self, mongoCollection, mongoCollectionName):
        
        missingCoinIdentifiers = mongoCollection.find({"currencies": {"$exists": False}})
        bulkUpdatedData = []
        
        for doc in missingCoinIdentifiers:
            docId = doc.get('_id')
            title = doc.get('title', '')
            text = doc.get('text', '')
            
            extractedCoins=[]
            combinedContent = f"{title}\n{text}".strip()

            if mongoCollectionName == "articles":
                extractedCoins = self.extractMentionedCoins(combinedContent)
                
            elif mongoCollectionName == "reddit":
                subreddit = doc.get('subreddit', '')
                extractedCoins = self.extractMentionedCoins(combinedContent, source=subreddit)   
            
            else:
                raise Exception(f"Invalid mongo collection {mongoCollectionName}")
            
            # Create the sentiment field to update
            bulkUpdatedData.append(
                {
                    "docId": docId,
                    "currencies": extractedCoins
                }
            )
            
        # Save all fetched data to MongoDB
        self.saveToMongo(bulkUpdatedData, mongoCollection)
        if self.SHOW_LOGS: print(f"Finished updating {len(bulkUpdatedData)} document(s) using coin identifier.")
        

    def extractMentionedCoins(self, combinedContent, source=None):
        # If subreddit is about a coin, by default add it
        if source:
            combinedContent += f" {source}"

        contentLower = combinedContent.lower()
        mentionedTickers = set()

        for ticker, fullName in self.allTopCoins.items():
            tickerPattern = rf'\b{re.escape(ticker.lower())}\b'
            fullNamePattern = rf'\b{re.escape(fullName.lower())}\b'

            if re.search(tickerPattern, contentLower) or re.search(fullNamePattern, contentLower):
                mentionedTickers.add(ticker)

        return list(mentionedTickers)


    def saveToMongo(self, data, mongoCollection):
        # Use upserts to avoid duplicates
        operations = [
            UpdateOne(
                {"_id": item["docId"]},
                {"$set": {"currencies": item["currencies"]}}
            )
            for item in data
        ]
        
        if operations:
            mongoCollection.bulk_write(operations)

