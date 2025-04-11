import os
from bson import CodecOptions
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

class SentimentAnalysis:
    def __init__(self, SHOW_LOGS=True):
        load_dotenv()
        
        self.SHOW_LOGS=SHOW_LOGS
        mongoDBURI = os.getenv("MONGODB_URI")
        if not mongoDBURI:
            raise ValueError("Please set the MONGODB_URI environment variable first.")

        # Database connection
        mongoClient = MongoClient(mongoDBURI)
        
        mongoCollectionReddit = mongoClient['ASM'].get_collection('reddit', codec_options=CodecOptions(tz_aware=True))
        mongoCollectionArticles = mongoClient['ASM'].get_collection('articles', codec_options=CodecOptions(tz_aware=True))
        self.allMongoCollections = [mongoCollectionReddit, mongoCollectionArticles]


    def analyzeSentimentsForAllCollections(self):
        for mongoCollection in self.allMongoCollections:
            if self.SHOW_LOGS: print("Sentiment analysis for collection:", mongoCollection.name)
            self.analyzeSentimentsForCollection(mongoCollection)
            if self.SHOW_LOGS: print("\n")
        

    def analyzeSentimentsForCollection(self, mongoCollection):
        
        # Query all documents that do not have the "sentiment" field
        missingSentiments = mongoCollection.find({"sentiment": {"$exists": False}})
        bulkUpdatedData = []

        for doc in missingSentiments:
            docId = doc.get('_id')
            title = doc.get('title', '')
            text = doc.get('text', '')

            combinedContent = f"{title}\n{text}".strip()

            sentiment, scores = self.analyzeSentimentForEntry(combinedContent)

            # Create the sentiment field to update
            sentimentData = {
                "label": sentiment,
                "scores": scores
            }
            
            bulkUpdatedData.append(
                {
                    "docId": docId,
                    "sentiment": sentimentData
                }
            )

        # Save all fetched data to MongoDB
        self.saveToMongo(bulkUpdatedData, mongoCollection)
        if self.SHOW_LOGS: print(f"Finished updating {len(bulkUpdatedData)} document(s) with sentiment analysis.")
        
        
    def analyzeSentimentForEntry(self, content):
        # Initialize VADER
        analyzer = SentimentIntensityAnalyzer()

        # Get sentiment scores
        scores = analyzer.polarity_scores(content)

        # Determine overall sentiment based on the compound score
        compound_score = scores['compound']
        if compound_score >= 0.05:
            sentiment = 'Positive'
        elif compound_score <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return sentiment, scores


    def saveToMongo(self, data, mongoCollection):
        # Use upserts to avoid duplicates
        operations = [
            UpdateOne(
                {"_id": item["docId"]},
                {"$set": {"sentiment": item["sentiment"]}}
            )
            for item in data
        ]
        
        if operations:
            mongoCollection.bulk_write(operations)


if __name__ == "__main__":
    sentimentAnalysis = SentimentAnalysis()
    sentimentAnalysis.analyzeSentimentsForAllCollections()
