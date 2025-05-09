import json
import os
from bson import CodecOptions
import feedparser
from datetime import datetime
from bs4 import BeautifulSoup
from pymongo import MongoClient, UpdateOne
from email.utils import parsedate_to_datetime

class Articles:
    
    def __init__(self, SHOW_LOGS=True):
        self.SHOW_LOGS=SHOW_LOGS
        mongoDBURI = os.getenv("MONGODB_URI")
        
        if not mongoDBURI:
            raise ValueError("Please set the mongoDB environment variable first.")

        # Database connection
        mongoClient = MongoClient(mongoDBURI)
        self.mongoCollection = mongoClient['ASM'].get_collection('articles', codec_options=CodecOptions(tz_aware=True))

        self.allWebsiteArticles=self.getAllWebsiteArticles()
    
    
    def fetchAllWebsiteArticlesContent(self):
        for data in self.allWebsiteArticles:
            allArticlesFromwebsite = self.fetchArticlesFromWebsite(data["journal"], data["url"])
            self.saveToMongo(allArticlesFromwebsite)
            if self.SHOW_LOGS: print(f"Fetched data from {data['journal']}")
    
    
    def fetchArticlesFromWebsite(self, websiteName, websiteURL):
        feed = feedparser.parse(websiteURL)
        allArticlesFromURL= [
            {
                "source": websiteName,
                "title": entry.title,
                "link": entry.link,
                "date": self.parseDate(entry.get("published")),
                "text": self.stripHtml(entry.get("summary", "")),
            }
            for entry in feed.entries
        ]
        return allArticlesFromURL


    def stripHtml(self, html):
        if not html or "<" not in html:
            return html.strip()
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove unwanted tags completely
        for tag in soup(["script", "style", "figure"]):
            tag.decompose()
            
        return soup.get_text(separator=" ", strip=True)
    
    
    def saveToMongo(self, data):
        # Use upserts to avoid duplicates
        operations = [
            UpdateOne({"title": item["title"]}, {"$set": item}, upsert=True)
            for item in data
        ]
        if operations:
            self.mongoCollection.bulk_write(operations)
          
          
    def getAllWebsiteArticles(self):
        with open("Services/utils/allArticles.json", "r") as file:
            data = json.load(file)
        return data


    def parseDate(self, date_str):
        try:
            dt = parsedate_to_datetime(date_str)
            return dt if dt.tzinfo else dt.replace(tzinfo=datetime.timezone.utc)
        except Exception:
            return None
        
