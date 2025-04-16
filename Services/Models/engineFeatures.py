import pandas as pd
import numpy as np


def engineFeatures(self, processedData):
    # Check if we have price data (required)
    if processedData["price_df"].empty:
        print("No price data available for feature engineering")
        return pd.DataFrame()
    
    # Start with the price data as our base
    priceDF = processedData["price_df"].copy()
    
    # Ensure price data index has consistent timezone handling (convert to tz-naive)
    priceDF.index = priceDF.index.tz_localize(None)
    
    # Create a DataFrame for combined features
    featuresDF = pd.DataFrame(index=priceDF.index)
    
    # Add basic price features
    featuresDF['open'] = priceDF['open']
    featuresDF['high'] = priceDF['high']
    featuresDF['low'] = priceDF['low']
    featuresDF['close'] = priceDF['close']
    featuresDF['volume'] = priceDF['volumefrom']
    
    # Add technical indicators - Moving Averages
    featuresDF['ma5'] = priceDF['close'].rolling(window=5).mean()
    featuresDF['ma7'] = priceDF['close'].rolling(window=7).mean()
    featuresDF['ma14'] = priceDF['close'].rolling(window=14).mean()
    
    # Add Exponential Moving Averages (EMAs)
    featuresDF['ema5'] = priceDF['close'].ewm(span=5, adjust=False).mean()
    featuresDF['ema14'] = priceDF['close'].ewm(span=14, adjust=False).mean()
    
    # Add MA crossovers (signals)
    featuresDF['ma_crossover'] = (featuresDF['ma5'] > featuresDF['ma14']).astype(int)
    
    # Add price changes
    featuresDF['price_change_1d'] = priceDF['close'].pct_change(1)
    featuresDF['price_change_3d'] = priceDF['close'].pct_change(3)
    featuresDF['price_change_7d'] = priceDF['close'].pct_change(7)
    
    # Add volatility measures
    featuresDF['volatility_5d'] = priceDF['close'].rolling(window=5).std()
    featuresDF['volatility_ratio'] = (featuresDF['volatility_5d'] / priceDF['close']).replace([np.inf, -np.inf], 0)
    
    # Add Bollinger Bands
    std_20 = priceDF['close'].rolling(window=20).std()
    featuresDF['bb_upper'] = featuresDF['ma14'] + (std_20 * 2)
    featuresDF['bb_lower'] = featuresDF['ma14'] - (std_20 * 2)
    featuresDF['bb_width'] = (featuresDF['bb_upper'] - featuresDF['bb_lower']) / featuresDF['ma14']
    
    # Calculate RSI (Relative Strength Index)
    delta = priceDF['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    featuresDF['rsi'] = 100 - (100 / (1 + rs))
    
    # Price momentum
    featuresDF['momentum'] = priceDF['close'] - priceDF['close'].shift(5)
    
    # Add fear and greed index if available
    if not processedData["fear_greed_df"].empty:
        fearGreedDF = processedData["fear_greed_df"].copy()
        
        # Ensure fear greed data index has consistent timezone handling
        fearGreedDF.index = fearGreedDF.index.tz_localize(None)
        
        # Reindex fear_greed data to match the price data dates
        # This handles cases where dates don't perfectly align
        fearGreedDF = fearGreedDF.reindex(featuresDF.index, method='ffill')
        
        # Add value directly
        featuresDF['fear_greed_index'] = fearGreedDF['value']
        
        # Add binary indicators for extreme sentiments
        featuresDF['is_extreme_fear'] = (fearGreedDF['classification'] == 'Extreme Fear').astype(int)
        featuresDF['is_extreme_greed'] = (fearGreedDF['classification'] == 'Extreme Greed').astype(int)
    
    # Add sentiment data if available
    if "sentiment_df" in processedData and not processedData["sentiment_df"].empty:
        sentimentDF = processedData["sentiment_df"].copy()
        
        # Ensure date column is datetime
        sentimentDF['date'] = pd.to_datetime(sentimentDF['date'])
        
        # Remove timezone info to be consistent with other datetime data
        sentimentDF['date'] = sentimentDF['date'].dt.tz_localize(None)
        
        # Group sentiment by day
        daily_sentiment = sentimentDF.groupby(sentimentDF['date'].dt.date).agg({
            'sentiment_score': 'mean',
            'source': 'count',
            'sentiment_label': lambda x: x.value_counts().index[0] if len(x) > 0 else 'Neutral'
        }).rename(columns={
            'sentiment_score': 'avg_sentiment_score',
            'source': 'sentiment_count'
        })
        
        # Convert index to datetime and ensure no timezone info
        daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
        
        # Calculate the percentage of positive, negative, and neutral sentiments
        sentiment_counts = sentimentDF.groupby(sentimentDF['date'].dt.date)['sentiment_label'].value_counts().unstack().fillna(0)
        
        # Calculate percentages only if we have sentiment data
        total_counts = sentiment_counts.sum(axis=1)
        
        for category in ['Positive', 'Negative', 'Neutral']:
            if category in sentiment_counts.columns:
                sentiment_counts[f'pct_{category.lower()}'] = sentiment_counts[category] / total_counts
            else:
                sentiment_counts[f'pct_{category.lower()}'] = 0
        
        # Convert index to datetime with no timezone info
        sentiment_counts.index = pd.to_datetime(sentiment_counts.index)
        
        # Join sentiment metrics with features using safe join
        for col in ['pct_positive', 'pct_negative', 'pct_neutral']:
            if col in sentiment_counts.columns:
                # Create temp series with the right index to match featuresDF
                temp_series = pd.Series(
                    sentiment_counts[col].values,
                    index=sentiment_counts.index
                )
                # Reindex to match featuresDF index
                reindexed_series = temp_series.reindex(
                    featuresDF.index, method='ffill'
                )
                featuresDF[col] = reindexed_series
        
        # Join average sentiment score using the same approach
        for col in ['avg_sentiment_score', 'sentiment_count']:
            if col in daily_sentiment.columns:
                temp_series = pd.Series(
                    daily_sentiment[col].values,
                    index=daily_sentiment.index
                )
                reindexed_series = temp_series.reindex(
                    featuresDF.index, method='ffill'
                )
                featuresDF[col] = reindexed_series
        
        # Fill missing values with neutral values
        if 'avg_sentiment_score' in featuresDF.columns:
            featuresDF['avg_sentiment_score'] = featuresDF['avg_sentiment_score'].fillna(0)
        if 'pct_positive' in featuresDF.columns:
            featuresDF['pct_positive'] = featuresDF['pct_positive'].fillna(0.33)
        if 'pct_negative' in featuresDF.columns:
            featuresDF['pct_negative'] = featuresDF['pct_negative'].fillna(0.33)
        if 'pct_neutral' in featuresDF.columns:
            featuresDF['pct_neutral'] = featuresDF['pct_neutral'].fillna(0.34)
        if 'sentiment_count' in featuresDF.columns:
            featuresDF['sentiment_count'] = featuresDF['sentiment_count'].fillna(0)
    
    # Forward fill any remaining NaN values
    featuresDF = featuresDF.ffill()
    
    # Drop NaN values that may have been introduced at the beginning by rolling operations
    featuresDF = featuresDF.dropna()
    
    return featuresDF
