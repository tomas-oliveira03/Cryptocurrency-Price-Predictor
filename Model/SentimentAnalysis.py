import os
from bson import CodecOptions
from dotenv import load_dotenv
from pymongo import MongoClient
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

class SentimentAnalysis:
    def __init__(self):
        load_dotenv()
        
        mongoDBURI = os.getenv("MONGODB_URI")
        if not mongoDBURI:
            raise ValueError("Please set the MONGODB_URI environment variable first.")

        # Database connection
        mongoClient = MongoClient(mongoDBURI)
        self.mongoCollection = mongoClient['ASM'].get_collection('reddit', codec_options=CodecOptions(tz_aware=True))


    def analyzeSentiment(self, content):
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


    def analyzeAndStoreMissingSentiments(self):
        
        # Query all documents that do not have the "sentiment" field
        missingSentiments = self.mongoCollection.find({"sentiment": {"$exists": False}})
        updatedCount = 0

        for doc in missingSentiments:
            docId = doc.get('_id')
            title = doc.get('title', '')
            text = doc.get('text', '')

            combinedContent = f"{title}\n{text}".strip()

            sentiment, scores = self.analyzeSentiment(combinedContent)

            # Create the sentiment field to update
            sentiment_data = {
                "label": sentiment,
                "scores": scores
            }

            # Update the document in MongoDB
            self.mongoCollection.update_one(
                {"_id": docId},
                {"$set": {"sentiment": sentiment_data}}
            )

            print(f"Updated document with _id {docId} → Sentiment: {sentiment}")
            updatedCount += 1

        print(f"\nFinished updating {updatedCount} document(s) with sentiment analysis.")


if __name__ == "__main__":
    sentimentAnalysis = SentimentAnalysis()
    sentimentAnalysis.analyzeAndStoreMissingSentiments()
