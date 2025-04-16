import pandas as pd

def preprocessData(self, rawData):
    processedData = {}
    
    # Process price data
    if rawData["price_data"]:
        priceDF = pd.DataFrame(rawData["price_data"])
        # _id is already excluded in the fetchData function
        priceDF['date'] = pd.to_datetime(priceDF['date'])
        priceDF.set_index('date', inplace=True)
        priceDF.sort_index(inplace=True)
        processedData["price_df"] = priceDF
    else:
        processedData["price_df"] = pd.DataFrame()
        
    # Process fear and greed data
    if rawData["fear_greed_data"]:
        fearGreedDF = pd.DataFrame(rawData["fear_greed_data"])
        # _id is already excluded in the fetchData function
        fearGreedDF['date'] = pd.to_datetime(fearGreedDF['date'])
        fearGreedDF.set_index('date', inplace=True)
        fearGreedDF.sort_index(inplace=True)
        processedData["fear_greed_df"] = fearGreedDF
    else:
        processedData["fear_greed_df"] = pd.DataFrame()
    
    # Merge Reddit, Forum, and Articles data into a single DataFrame
    # Initialize an empty list to hold all sentiment data
    all_sentiment_data = []
    
    # Process Reddit data
    if rawData["reddit_data"]:
        redditDF = pd.DataFrame(rawData["reddit_data"])
        # Add source identifier
        redditDF['source'] = 'reddit'
        # Rename created_at to date for consistency
        redditDF = redditDF.rename(columns={'created_at': 'date'})
        all_sentiment_data.append(redditDF)
    
    # Process Forum data
    if rawData["forum_data"]:
        forumDF = pd.DataFrame(rawData["forum_data"])
        # Add source identifier
        forumDF['source'] = 'forum'
        # Rename created_at to date for consistency
        forumDF = forumDF.rename(columns={'created_at': 'date'})
        all_sentiment_data.append(forumDF)
    
    # Process Articles data (if available)
    if rawData["articles_data"]:
        articlesDF = pd.DataFrame(rawData["articles_data"])
        # Add source identifier
        articlesDF['source'] = 'articles'
        # date field already named correctly
        all_sentiment_data.append(articlesDF)
    
    # Merge all sentiment data if available
    if all_sentiment_data:
        # Concatenate all dataframes
        mergedSentimentDF = pd.concat(all_sentiment_data, ignore_index=True)
        
        # Convert date to datetime
        mergedSentimentDF['date'] = pd.to_datetime(mergedSentimentDF['date'])
        
        # Extract sentiment information
        if 'sentiment' in mergedSentimentDF.columns:
            try:
                # Extract sentiment label
                mergedSentimentDF['sentiment_label'] = mergedSentimentDF['sentiment'].apply(
                    lambda x: x.get('label', 'Neutral') if isinstance(x, dict) else 'Neutral'
                )
                
                # Extract sentiment compound score
                mergedSentimentDF['sentiment_score'] = mergedSentimentDF['sentiment'].apply(
                    lambda x: x.get('scores', {}).get('compound', 0) if isinstance(x, dict) else 0
                )
                
                # Drop the original sentiment column as we've extracted what we need
                mergedSentimentDF.drop('sentiment', axis=1, inplace=True)
            except Exception as e:
                print(f"Error extracting sentiment: {e}")
        
        # Sort by date
        mergedSentimentDF.sort_values('date', inplace=True)
        
        # Add to processed data
        processedData["sentiment_df"] = mergedSentimentDF
    else:
        processedData["sentiment_df"] = pd.DataFrame()
    
    return processedData
