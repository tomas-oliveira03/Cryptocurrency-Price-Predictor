from datetime import datetime, timedelta

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

    # Fetch data from numberOfPastDaysOfData ago up to today
    startDate = datetime.now() - timedelta(days=numberOfPastDaysOfData)
    endDate = datetime.now() 

    print(f"Fetching data from {startDate.strftime('%Y-%m-%d')} to {endDate.strftime('%Y-%m-%d')}")

    # Get data from all sources using the date range
    cryptoData = getCryptoPriceData(self, startDate=startDate, endDate=endDate, cryptoSymbol=cryptoCoin)
    fearGreedData = getFearGreedData(self, startDate=startDate, endDate=endDate)

    # Get social media data related to this cryptocurrency
    redditData = getRedditData(self, startDate=startDate, endDate=endDate, cryptoSymbol=cryptoCoin)
    forumData = getForumData(self, startDate=startDate, endDate=endDate, cryptoSymbol=cryptoCoin)
    articlesData = getArticlesData(self, startDate=startDate, endDate=endDate, cryptoSymbol=cryptoCoin)

    datasets = {
        "price_data": cryptoData,
        "fear_greed_data": fearGreedData,
        "reddit_data": redditData,
        "forum_data": forumData,
        "articles_data": articlesData
    }

    return datasets