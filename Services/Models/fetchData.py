from datetime import datetime, timedelta

import pandas as pd


def getFearGreedData(self, startDate=None, endDate=None):
    query = {}
    
    # Add date filters if provided
    if startDate or endDate:
        query["date"] = {}
        if startDate:
            query["date"]["$gte"] = startDate
        if endDate:
            query["date"]["$lte"] = endDate
    
    # Define fields to return
    projection = {
        "_id": 0, 
        "date": 1,
        "classification": 1,
        "value": 1
    }
    
    # Execute query with projection and return results
    results = list(self.cryptoFearGreedDB.find(query, projection).sort("date", -1))
    return results


def getCryptoPriceData(self, startDate=None, endDate=None, cryptoSymbol=None):
    query = {}
    
    if cryptoSymbol:
        query["cryptoCurrency"] = cryptoSymbol
    
    # Add date filters if provided
    if startDate or endDate:
        query["date"] = {}
        if startDate:
            query["date"]["$gte"] = startDate
        if endDate:
            query["date"]["$lte"] = endDate
    
    # Define fields to return
    projection = {
        "_id": 0, 
        "date": 1,
        "averagePrice": 1,
        "close": 1,
        "high": 1,
        "low": 1,
        "open": 1,
        "volumefrom": 1,
        "volumeto": 1
    }
    
    # Execute query and return results
    results = list(self.detailedCryptoData.find(query, projection).sort("date", -1))
    return results




def getRedditData(self, startDate=None, endDate=None, cryptoSymbol=None):
    query = {}

    # Add cryptocurrency filter if provided
    if cryptoSymbol:
        query["currencies"] = cryptoSymbol

    # Add date filters if provided
    if startDate or endDate:
        query["created_at"] = {}
        if startDate:
            query["created_at"]["$gte"] = startDate
        if endDate:
            query["created_at"]["$lte"] = endDate

    # Define fields to return
    projection = {
        "_id": 0,
        "created_at": 1,
        "sentiment": 1
    }

    # Execute query with projection and return sorted results
    results = list(self.redditDB.find(query, projection).sort("created_at", -1))
    return results


def getForumData(self, startDate=None, endDate=None, cryptoSymbol=None):
    query = {}
    
    # Add cryptocurrency filter if provided
    if cryptoSymbol:
        query["currencies.code"] = cryptoSymbol
    
    # Add date filters if provided
    if startDate or endDate:
        query["created_at"] = {}
        if startDate:
            query["created_at"]["$gte"] = startDate
        if endDate:
            query["created_at"]["$lte"] = endDate

    # Define fields to return
    projection = {
        "_id": 0,
        "created_at": 1,
        "sentiment": 1
    }

    # Execute query with projection and return sorted results
    results = list(self.forumDB.find(query, projection).sort("created_at", -1))
    return results


def getArticlesData(self, startDate=None, endDate=None, cryptoSymbol=None):
    query = {}
    
    # Add cryptocurrency filter if provided
    if cryptoSymbol:
        query["currencies"] = cryptoSymbol
    
    # Add date filters if provided
    if startDate or endDate:
        query["date"] = {}
        if startDate:
            query["date"]["$gte"] = startDate
        if endDate:
            query["date"]["$lte"] = endDate

    # Define fields to return
    projection = {
        "_id": 0,
        "date": 1,
        "sentiment": 1,
    }

    # Execute query with projection and return sorted results
    results = list(self.articlesDB.find(query, projection).sort("date", -1))
    return results






def getDataForPrediction(self, cryptoCoin=None, numberOfPastDaysOfData=30):       

    startDate = datetime.now() - timedelta(days=numberOfPastDaysOfData)
    endDate = datetime.now() - timedelta(days=5)
    
    # Get data from all sources
    cryptoData = getCryptoPriceData(self, startDate=startDate, cryptoSymbol=cryptoCoin)
    fearGreedData = getFearGreedData(self, startDate=startDate)
    
    # Get social media data related to this cryptocurrency
    redditData = getRedditData(self, startDate=startDate, cryptoSymbol=cryptoCoin)
    forumData = getForumData(self, startDate=startDate, cryptoSymbol=cryptoCoin)
    articlesData = getArticlesData(self, startDate=startDate, cryptoSymbol=cryptoCoin)

    datasets = {
        "price_data": cryptoData,
        "fear_greed_data": fearGreedData,
        "reddit_data": redditData,
        "forum_data": forumData,
        "articles_data": articlesData
    }
    
    
    
    # Print the number of entries in each key of datasets
    print("Raw data summary:")
    total_price_data = min(len(datasets.get("price_data", [])), len(datasets.get("fear_greed_data", [])))
    data_count = sum(len(datasets[key]) for key in ["reddit_data", "forum_data", "articles_data"] if key in datasets and datasets[key] is not None)
    
    for key, value in datasets.items():
        if isinstance(value, pd.DataFrame):
            print(f"  {key}: {len(value)} entries")
        else:
            print(f"  {key}: {len(value) if hasattr(value, '__len__') else 'N/A'} entries")
    
    print(f"\nTotal price data (min of price_data and fear_greed_data): {total_price_data}")
    print(f"Data count (sum of reddit, forum, and articles): {data_count}\n")
    
        

    return datasets