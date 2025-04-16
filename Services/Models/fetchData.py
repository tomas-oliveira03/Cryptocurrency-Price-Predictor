def getFearGreedData(self, startDate=None, endDate=None):
    query = {}
    
    # Add date filters if provided
    if startDate or endDate:
        query["date"] = {}
        if startDate:
            query["date"]["$gte"] = startDate
        if endDate:
            query["date"]["$lte"] = endDate
    
    # Execute query and return results
    results = list(self.cryptoFearGreedDB.find(query).sort("date", -1))
    return results


def getCryptoPriceData(self, cryptoSymbol=None, startDate=None, endDate=None):
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
    
    # Execute query and return results
    results = list(self.detailedCryptoData.find(query).sort("date", -1))
    return results


def getRedditData(self, subreddit=None, startDate=None, endDate=None):
    query = {}

    # Add subreddit filter if provided
    if subreddit:
        query["subreddit"] = subreddit

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


def getForumData(self, cryptoSymbol=None, startDate=None, endDate=None):
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
        "currencies": 1,
        "source": 1,
        "votes": 1,
        "sentiment": 1
    }

    # Execute query with projection and return sorted results
    results = list(self.forumDB.find(query, projection).sort("created_at", -1))
    return results

