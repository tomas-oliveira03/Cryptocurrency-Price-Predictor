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
    
    # Execute query and return results
    results = list(self.redditDB.find(query).sort("created_at", -1))
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
    
    # Execute query and return results
    results = list(self.forumDB.find(query).sort("created_at", -1))
    return results

