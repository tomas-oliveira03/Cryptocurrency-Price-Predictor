import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pymongo
from typing import Dict, Tuple


class CryptoPricePredictor:
    def __init__(self, db_connection_string: str = "mongodb://localhost:27017/", 
                 db_name: str = "ASM"):
        """
        Initialize the CryptoPricePredictor with MongoDB connection
        
        Args:
            db_connection_string: MongoDB connection string
            db_name: Name of the database
        """
        self.client = pymongo.MongoClient(db_connection_string)
        self.db = self.client[db_name]
        self.models = {
            'xgboost': None,
            'random_forest': None,
            'lstm': None
        }
        self.best_model = None
        self.price_scaler = MinMaxScaler()
        self.features_scaler = MinMaxScaler()
        self.lookback_days = 14  # Number of past days to consider
        
    def load_data(self, crypto_symbol: str = "BTC", 
                  start_date: datetime = None, 
                  end_date: datetime = None) -> Dict[str, pd.DataFrame]:
        """
        Load data from all collections for a specific cryptocurrency
        
        Args:
            crypto_symbol: Symbol of the cryptocurrency to predict
            start_date: Start date for data loading
            end_date: End date for data loading
            
        Returns:
            Dictionary containing DataFrames for each data source
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)  # Default to one year
        if not end_date:
            end_date = datetime.now()
        
        # Load crypto price data
        price_query = {
            "cryptoCurrency": crypto_symbol,
            "date": {"$gte": start_date, "$lte": end_date}
        }
        price_data = list(self.db["detailed-crypto-data"].find(price_query))
        price_df = pd.DataFrame(price_data)
        
        # Load fear & greed index
        fear_greed_query = {
            "date": {"$gte": start_date, "$lte": end_date}
        }
        fear_greed_data = list(self.db["crypto-fear-greed"].find(fear_greed_query))
        fear_greed_df = pd.DataFrame(fear_greed_data)
        
        # Load news articles with sentiment
        articles_query = {
            "date": {"$gte": start_date, "$lte": end_date}
        }
        articles_data = list(self.db["articles"].find(articles_query))
        articles_df = pd.DataFrame(articles_data)
        
        # Load Reddit posts with sentiment
        reddit_query = {
            "created_at": {"$gte": start_date, "$lte": end_date}
        }
        reddit_data = list(self.db["reddit"].find(reddit_query))
        reddit_df = pd.DataFrame(reddit_data)
        
        return {
            "price": price_df,
            "fear_greed": fear_greed_df,
            "articles": articles_df,
            "reddit": reddit_df
        }
    
    def preprocess_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Preprocess and merge all datasets into a single DataFrame with features
        
        Args:
            data_dict: Dictionary containing DataFrames for each data source
            
        Returns:
            DataFrame with processed features
        """
        # Preprocess price data
        price_df = data_dict["price"]
        if price_df.empty:
            raise ValueError("No price data found for the specified period.")
        
        # Convert date to datetime if it's not
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df.set_index('date', inplace=True)
        price_df.sort_index(inplace=True)
        
        # Create basic price features
        df = pd.DataFrame()
        df['open'] = price_df['open']
        df['high'] = price_df['high']
        df['low'] = price_df['low']
        df['close'] = price_df['close']
        df['volume'] = price_df['volumefrom']
        
        # Add technical indicators
        df['ma7'] = df['close'].rolling(window=7).mean()
        df['ma14'] = df['close'].rolling(window=14).mean()
        df['ma30'] = df['close'].rolling(window=30).mean()
        
        # Price change percentages
        df['price_change_1d'] = df['close'].pct_change(1)
        df['price_change_3d'] = df['close'].pct_change(3)
        df['price_change_7d'] = df['close'].pct_change(7)
        
        # Volatility measures
        df['volatility_7d'] = df['close'].rolling(window=7).std()
        
        # Process fear and greed data
        if not data_dict["fear_greed"].empty:
            fear_greed_df = data_dict["fear_greed"]
            fear_greed_df['date'] = pd.to_datetime(fear_greed_df['date'])
            fear_greed_df.set_index('date', inplace=True)
            fear_greed_df.sort_index(inplace=True)
            
            # Add fear & greed index to main dataframe
            df = df.join(fear_greed_df['value'].rename('fear_greed_index'), how='left')
            df['fear_greed_index'] = df['fear_greed_index'].ffill()
        
        # Process sentiment data from articles
        if not data_dict["articles"].empty:
            articles_df = data_dict["articles"]
            articles_df['date'] = pd.to_datetime(articles_df['date'])
            
            # Check the structure of the sentiment data
            print("Processing articles sentiment data...")
            
            # Check if sentiment is a nested structure or columns with dots
            if 'sentiment' in articles_df.columns and isinstance(articles_df['sentiment'].iloc[0], dict):
                # Extract sentiment fields from the nested dictionary
                articles_df['compound'] = articles_df['sentiment'].apply(lambda x: x.get('scores', {}).get('compound', 0))
                articles_df['pos'] = articles_df['sentiment'].apply(lambda x: x.get('scores', {}).get('pos', 0))
                articles_df['neg'] = articles_df['sentiment'].apply(lambda x: x.get('scores', {}).get('neg', 0))
                articles_df['neu'] = articles_df['sentiment'].apply(lambda x: x.get('scores', {}).get('neu', 0))
                
                # Aggregate sentiment by day
                daily_sentiment = articles_df.groupby(articles_df['date'].dt.date).agg({
                    'compound': 'mean',
                    'pos': 'mean',
                    'neg': 'mean',
                    'neu': 'mean',
                    '_id': 'count'
                }).rename(columns={
                    'compound': 'news_sentiment_compound',
                    'pos': 'news_sentiment_pos',
                    'neg': 'news_sentiment_neg',
                    'neu': 'news_sentiment_neu',
                    '_id': 'news_count'
                })
            else:
                # Print available columns for debugging
                print("Available columns in articles_df:", articles_df.columns.tolist())
                
                # Try to find sentiment columns that contain the word "sentiment"
                sentiment_cols = [col for col in articles_df.columns if 'sentiment' in col.lower()]
                print("Found sentiment columns:", sentiment_cols)
                
                # If no sentiment columns are found, use a simplified approach
                daily_sentiment = articles_df.groupby(articles_df['date'].dt.date).agg({
                    '_id': 'count'
                }).rename(columns={'_id': 'news_count'})
            
            daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
            df = df.join(daily_sentiment, how='left')
        
        # Process sentiment data from Reddit
        if not data_dict["reddit"].empty:
            reddit_df = data_dict["reddit"]
            reddit_df['created_at'] = pd.to_datetime(reddit_df['created_at'])
            
            # Check the structure of the sentiment data
            print("Processing reddit sentiment data...")
            
            # Check if sentiment is a nested structure or columns with dots
            if 'sentiment' in reddit_df.columns and isinstance(reddit_df['sentiment'].iloc[0], dict):
                # Extract sentiment fields from the nested dictionary
                reddit_df['compound'] = reddit_df['sentiment'].apply(lambda x: x.get('scores', {}).get('compound', 0))
                reddit_df['pos'] = reddit_df['sentiment'].apply(lambda x: x.get('scores', {}).get('pos', 0))
                reddit_df['neg'] = reddit_df['sentiment'].apply(lambda x: x.get('scores', {}).get('neg', 0))
                reddit_df['neu'] = reddit_df['sentiment'].apply(lambda x: x.get('scores', {}).get('neu', 0))
                
                # Aggregate Reddit sentiment by day
                reddit_daily = reddit_df.groupby(reddit_df['created_at'].dt.date).agg({
                    'compound': 'mean',
                    'pos': 'mean',
                    'neg': 'mean',
                    'neu': 'mean',
                    'score': 'mean',
                    'num_comments': 'sum',
                    '_id': 'count'
                }).rename(columns={
                    'compound': 'reddit_sentiment_compound',
                    'pos': 'reddit_sentiment_pos',
                    'neg': 'reddit_sentiment_neg',
                    'neu': 'reddit_sentiment_neu',
                    'score': 'avg_reddit_score',
                    'num_comments': 'total_comments',
                    '_id': 'reddit_post_count'
                })
            else:
                # Print available columns for debugging
                print("Available columns in reddit_df:", reddit_df.columns.tolist())
                
                # Try to find sentiment columns that contain the word "sentiment"
                sentiment_cols = [col for col in reddit_df.columns if 'sentiment' in col.lower()]
                print("Found sentiment columns:", sentiment_cols)
                
                # If no sentiment columns are found, use a simplified approach
                reddit_daily = reddit_df.groupby(reddit_df['created_at'].dt.date).agg({
                    'score': 'mean',
                    'num_comments': 'sum',
                    '_id': 'count'
                }).rename(columns={
                    'score': 'avg_reddit_score',
                    'num_comments': 'total_comments',
                    '_id': 'reddit_post_count'
                })
            
            reddit_daily.index = pd.to_datetime(reddit_daily.index)
            df = df.join(reddit_daily, how='left')
        
        # Fill missing values
        df = df.ffill()
        df.dropna(inplace=True)
        
        return df
    
    def create_features_targets(self, df: pd.DataFrame, 
                                target_column: str = 'close', 
                                forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create features and target arrays for model training
        
        Args:
            df: Preprocessed DataFrame with all features
            target_column: Column to predict
            forecast_horizon: Number of days ahead to predict
            
        Returns:
            X: Feature matrix
            y: Target array
        """
        # Create target: the price 'forecast_horizon' days in the future
        df[f'target_{forecast_horizon}d'] = df[target_column].shift(-forecast_horizon)
        
        # Drop rows with NaN targets
        df.dropna(inplace=True)
        
        # Select features and target
        features = df.drop(columns=[f'target_{forecast_horizon}d'])
        target = df[f'target_{forecast_horizon}d']
        
        # Scale features and target
        feature_names = features.columns
        features_scaled = self.features_scaler.fit_transform(features)
        target_scaled = self.price_scaler.fit_transform(target.values.reshape(-1, 1))
        
        return features_scaled, target_scaled.ravel(), feature_names
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray, 
                         seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM model
        
        Args:
            X: Feature matrix
            y: Target array
            seq_length: Length of sequence (lookback period)
            
        Returns:
            X_seq: Sequence features
            y_seq: Sequence targets
        """
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_and_train_models(self, df: pd.DataFrame, 
                               target_column: str = 'close', 
                               forecast_horizon: int = 1,
                               test_size: float = 0.2) -> Dict:
        """
        Build and train multiple models and select the best one
        
        Args:
            df: Preprocessed DataFrame with all features
            target_column: Column to predict
            forecast_horizon: Number of days ahead to predict
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with model performance metrics
        """
        # Create features and targets
        X, y, feature_names = self.create_features_targets(
            df, target_column, forecast_horizon
        )
        
        print(f"Total data points: {len(X)}")
        
        # Lower the minimum data requirement to 10
        min_data_points = 10
        if len(X) < min_data_points:
            raise ValueError(f"Not enough data points for training. Need at least {min_data_points} data points.")
        
        # For small datasets, use a smaller test set
        if len(X) < 30:
            test_size = 0.2 if len(X) >= 20 else 0.1
            print(f"Small dataset detected. Using test_size={test_size}")
            
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        print(f"Training data points: {len(X_train)}, Testing data points: {len(X_test)}")
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Add a simple moving average model as baseline
        print("Setting up moving average baseline...")
        # Save original scale data for later use
        y_test_orig = self.price_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # For small datasets, simplify the models to avoid overfitting
        is_small_dataset = len(X) < 30
        
        # Train XGBoost model with parameters adjusted for dataset size
        print("Training XGBoost model...")
        n_estimators = 50 if is_small_dataset else 100
        max_depth = 3 if is_small_dataset else 5
        
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=max_depth,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        y_pred_xgb = xgb_model.predict(X_test)
        
        # Train Random Forest model with parameters adjusted for dataset size
        print("Training Random Forest model...")
        n_estimators_rf = 50 if is_small_dataset else 100
        max_depth_rf = 5 if is_small_dataset else 10
        
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators_rf, 
            max_depth=max_depth_rf,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        y_pred_rf = rf_model.predict(X_test)
        
        # Calculate metrics for tree-based models
        y_pred_xgb_orig = self.price_scaler.inverse_transform(y_pred_xgb.reshape(-1, 1)).flatten()
        y_pred_rf_orig = self.price_scaler.inverse_transform(y_pred_rf.reshape(-1, 1)).flatten()
        
        # Add a simple moving average prediction as baseline
        # Use the last 7 days (or less if not enough data) as prediction
        ma_window = min(7, len(y_train))
        y_pred_ma = np.full_like(y_test, np.mean(y_train[-ma_window:]))
        y_pred_ma_orig = self.price_scaler.inverse_transform(y_pred_ma.reshape(-1, 1)).flatten()
        
        metrics['moving_avg'] = {
            'mse': mean_squared_error(y_test_orig, y_pred_ma_orig),
            'mae': mean_absolute_error(y_test_orig, y_pred_ma_orig),
            'r2': r2_score(y_test_orig, y_pred_ma_orig)
        }
        
        metrics['xgboost'] = {
            'mse': mean_squared_error(y_test_orig, y_pred_xgb_orig),
            'mae': mean_absolute_error(y_test_orig, y_pred_xgb_orig),
            'r2': r2_score(y_test_orig, y_pred_xgb_orig)
        }
        
        metrics['random_forest'] = {
            'mse': mean_squared_error(y_test_orig, y_pred_rf_orig),
            'mae': mean_absolute_error(y_test_orig, y_pred_rf_orig),
            'r2': r2_score(y_test_orig, y_pred_rf_orig)
        }
        
        # For small datasets or limited data, skip LSTM
        if len(X_train) <= 20:
            print("Dataset too small for LSTM model. Skipping LSTM training.")
            self.models['lstm'] = None
        else:
            # Create sequences for LSTM - only if we have enough data
            seq_length = min(self.lookback_days, len(X_train) // 3)  # Reduce sequence length for small datasets
            print(f"Using sequence length of {seq_length} for LSTM model")
            
            try:
                print("Training LSTM model...")
                X_seq_train, y_seq_train = self.create_sequences(X_train, y_train, seq_length)
                X_seq_test, y_seq_test = self.create_sequences(X_test, y_test, seq_length)
                
                print(f"LSTM sequence training samples: {len(X_seq_train)}")
                print(f"LSTM sequence testing samples: {len(X_seq_test)}")
                
                # Only proceed with LSTM if we have sufficient sequences
                if len(X_seq_train) > 5 and len(X_seq_test) > 0:
                    # Calculate batch size based on data size (to avoid running out of data)
                    batch_size = min(8, max(1, len(X_seq_train) // 2))
                    epochs = min(20, max(10, len(X_seq_train)))
                    
                    # Simpler LSTM for small datasets
                    units = 20 if is_small_dataset else 50
                    dropout_rate = 0.1 if is_small_dataset else 0.2
                    
                    # Train LSTM model
                    lstm_model = Sequential([
                        LSTM(units, activation='relu', return_sequences=True, 
                             input_shape=(seq_length, X.shape[1])),
                        Dropout(dropout_rate),
                        LSTM(units, activation='relu'),
                        Dropout(dropout_rate),
                        Dense(1)
                    ])
                    
                    lstm_model.compile(optimizer='adam', loss='mse')
                    lstm_model.fit(
                        X_seq_train, y_seq_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.1,
                        verbose=1
                    )
                    
                    self.models['lstm'] = lstm_model
                    y_pred_lstm = lstm_model.predict(X_seq_test).flatten()
                    
                    # Calculate LSTM metrics
                    y_lstm_test_orig = self.price_scaler.inverse_transform(y_seq_test.reshape(-1, 1)).flatten()
                    y_pred_lstm_orig = self.price_scaler.inverse_transform(y_pred_lstm.reshape(-1, 1)).flatten()
                    
                    metrics['lstm'] = {
                        'mse': mean_squared_error(y_lstm_test_orig, y_pred_lstm_orig),
                        'mae': mean_absolute_error(y_lstm_test_orig, y_pred_lstm_orig),
                        'r2': r2_score(y_lstm_test_orig, y_pred_lstm_orig)
                    }
                else:
                    print("Not enough sequence data for LSTM model. Skipping LSTM training.")
                    self.models['lstm'] = None
                
            except Exception as e:
                print(f"Error during LSTM training: {e}")
                print("Skipping LSTM model due to error.")
                self.models['lstm'] = None
            
        # Select best model based on MSE, only considering models that were successfully trained
        available_models = [model_name for model_name in metrics.keys()]
        if available_models:
            best_model_name = min(available_models, key=lambda x: metrics[x]['mse'])
            self.best_model = best_model_name
            print(f"Best model selected: {best_model_name}")
        else:
            # Fallback to a default model if none were successfully trained
            self.best_model = 'xgboost'  # Default to XGBoost as fallback
            print("No models had valid metrics. Defaulting to XGBoost.")
        
        # Store feature names for future use
        self.feature_names = feature_names
        
        return metrics
    
    def predict_future(self, df: pd.DataFrame, days_ahead: int = 7) -> pd.DataFrame:
        """
        Predict future prices for a number of days ahead
        
        Args:
            df: DataFrame with historical data
            days_ahead: Number of days to predict
            
        Returns:
            DataFrame with predictions
        """
        if not self.best_model or not self.models[self.best_model]:
            raise ValueError("No trained model available. Train models first.")
            
        print(f"Making predictions using {self.best_model} model")
        
        # Get latest data for prediction
        latest_data = df.copy().iloc[-self.lookback_days:]
        
        # Prepare container for predictions
        predictions = []
        dates = []
        last_date = df.index[-1]
        
        # Create a copy of the latest data for rolling predictions
        predict_df = latest_data.copy()
        
        for i in range(days_ahead):
            # Next date
            next_date = last_date + timedelta(days=i+1)
            dates.append(next_date)
            
            # Prepare features
            features = predict_df.iloc[-1:].drop(columns=['target_1d'], errors='ignore')
            scaled_features = self.features_scaler.transform(features)
            
            # Make prediction based on best model
            if self.best_model == 'lstm':
                # For LSTM, we need sequences
                seq = self.features_scaler.transform(predict_df.iloc[-self.lookback_days:])
                seq = seq.reshape(1, self.lookback_days, seq.shape[1])
                pred_scaled = self.models[self.best_model].predict(seq)[0][0]
            else:
                pred_scaled = self.models[self.best_model].predict(scaled_features)[0]
            
            # Convert prediction back to original scale
            prediction = self.price_scaler.inverse_transform([[pred_scaled]])[0][0]
            predictions.append(prediction)
            
            # Update prediction DataFrame for next iteration
            new_row = features.copy()
            new_row['close'] = prediction
            # Simple estimates for other price points
            new_row['open'] = prediction * 0.99
            new_row['high'] = prediction * 1.02
            new_row['low'] = prediction * 0.98
            
            # Create a new row with the next date as index
            new_row.index = [next_date]
            
            # Append to the prediction dataframe
            predict_df = pd.concat([predict_df, new_row])
        
        # Create prediction DataFrame
        prediction_df = pd.DataFrame({
            'date': dates,
            'predicted_price': predictions
        })
        prediction_df.set_index('date', inplace=True)
        
        return prediction_df
    
    def evaluate_model(self, df: pd.DataFrame, 
                       target_column: str = 'close',
                       forecast_horizon: int = 1,
                       plot: bool = True) -> Dict:
        """
        Evaluate model performance and generate plots
        
        Args:
            df: DataFrame with historical data
            target_column: Column to predict
            forecast_horizon: Number of days ahead to predict
            plot: Whether to generate performance plots
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Train models and get metrics
        metrics = self.build_and_train_models(
            df, target_column, forecast_horizon
        )
        
        if plot:
            # Create a plot to compare actual vs predicted values
            X, y, _ = self.create_features_targets(df, target_column, forecast_horizon)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            # Get predictions from best model
            if self.best_model == 'lstm':
                seq_length = self.lookback_days
                X_seq_test, y_seq_test = self.create_sequences(X_test, y_test, seq_length)
                y_pred = self.models[self.best_model].predict(X_seq_test).flatten()
                y_test = y_seq_test
            else:
                y_pred = self.models[self.best_model].predict(X_test)
            
            # Convert back to original scale
            y_test_orig = self.price_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_orig = self.price_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            
            # Create dates for x-axis
            if self.best_model == 'lstm':
                test_dates = df.index[-(len(y_test_orig)):]
            else:
                test_dates = df.index[-len(y_test_orig):]
            
            plt.figure(figsize=(12, 6))
            plt.plot(test_dates, y_test_orig, label='Actual')
            plt.plot(test_dates, y_pred_orig, label=f'Predicted ({self.best_model})')
            plt.title(f'Actual vs Predicted {target_column} ({forecast_horizon}-day Forecast)')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('model_evaluation.png')
            plt.close()
        
        return metrics
    
    def feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance for tree-based models
        
        Returns:
            DataFrame with feature importances
        """
        if not hasattr(self, 'feature_names'):
            raise ValueError("No feature names available. Train models first.")
        
        importances = None
        model_name = None
        
        # Try to get feature importance from available models
        if self.models['xgboost'] is not None:
            importances = self.models['xgboost'].feature_importances_
            model_name = 'XGBoost'
        elif self.models['random_forest'] is not None:
            importances = self.models['random_forest'].feature_importances_
            model_name = 'Random Forest'
        
        if importances is not None:
            # Create DataFrame with feature importances
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            return importance_df
        else:
            return pd.DataFrame({'Message': ['No tree-based model available for feature importance']})

    def get_all_cryptocurrencies(self) -> list:
        """
        Get a list of all cryptocurrencies available in the database
        
        Returns:
            List of cryptocurrency symbols
        """
        # Query for all unique cryptocurrency symbols
        unique_cryptos = self.db["detailed-crypto-data"].distinct("cryptoCurrency")
        return sorted(unique_cryptos)
    
    def batch_predict(self, days_ahead: int = 7) -> Dict[str, pd.DataFrame]:
        """
        Make predictions for all available cryptocurrencies
        
        Args:
            days_ahead: Number of days to predict ahead
            
        Returns:
            Dictionary mapping cryptocurrency symbols to their prediction DataFrames
        """
        all_cryptos = self.get_all_cryptocurrencies()
        print(f"Found {len(all_cryptos)} cryptocurrencies: {', '.join(all_cryptos)}")
        
        predictions = {}
        
        for crypto in all_cryptos:
            try:
                print(f"\nProcessing {crypto}...")
                # Load data for this cryptocurrency
                data = self.load_data(crypto_symbol=crypto)
                
                # Preprocess data
                processed_df = self.preprocess_data(data)
                
                # Check if we have enough data
                if len(processed_df) < 10:  # Minimum required data points
                    print(f"Not enough data for {crypto}. Skipping.")
                    continue
                
                # Train models
                metrics = self.build_and_train_models(
                    processed_df, target_column='close', forecast_horizon=1
                )
                
                # Make predictions
                future_predictions = self.predict_future(processed_df, days_ahead=days_ahead)
                predictions[crypto] = future_predictions
                
                # Reset the model state for the next cryptocurrency
                self.models = {
                    'xgboost': None,
                    'random_forest': None,
                    'lstm': None
                }
                self.best_model = None
                
            except Exception as e:
                print(f"Error processing {crypto}: {e}")
        
        return predictions
    
    def get_market_overview(self) -> Dict:
        """
        Generate an overall market sentiment score from 1-100
        
        Returns:
            Dictionary containing market overview metrics
        """
        all_cryptos = self.get_all_cryptocurrencies()
        print(f"Analyzing {len(all_cryptos)} cryptocurrencies for market overview...")
        
        # Initialize metrics
        price_changes = []
        volume_changes = []
        price_trends = []  # 1 for uptrend, 0 for neutral, -1 for downtrend
        
        # Get the latest fear and greed index
        try:
            latest_fear_greed = list(self.db["crypto-fear-greed"].find().sort("date", -1).limit(1))
            if latest_fear_greed:
                fear_greed_score = latest_fear_greed[0].get("value", 50)
                fear_greed_classification = latest_fear_greed[0].get("classification", "Neutral")
            else:
                fear_greed_score = 50
                fear_greed_classification = "Neutral (No data)"
        except Exception as e:
            print(f"Error fetching fear & greed data: {e}")
            fear_greed_score = 50
            fear_greed_classification = "Neutral (Error)"
        
        # Collect news and social media sentiment
        try:
            # Get recent news sentiment
            one_week_ago = datetime.now() - timedelta(days=7)
            recent_articles = list(self.db["articles"].find(
                {"date": {"$gte": one_week_ago}}
            ))
            
            article_sentiments = []
            for article in recent_articles:
                if "sentiment" in article and "scores" in article["sentiment"]:
                    article_sentiments.append(article["sentiment"]["scores"].get("compound", 0))
            
            avg_news_sentiment = sum(article_sentiments) / max(1, len(article_sentiments))
            
            # Get recent Reddit sentiment
            recent_reddit = list(self.db["reddit"].find(
                {"created_at": {"$gte": one_week_ago}}
            ))
            
            reddit_sentiments = []
            for post in recent_reddit:
                if "sentiment" in post and "scores" in post["sentiment"]:
                    reddit_sentiments.append(post["sentiment"]["scores"].get("compound", 0))
            
            avg_reddit_sentiment = sum(reddit_sentiments) / max(1, len(reddit_sentiments))
            
        except Exception as e:
            print(f"Error analyzing sentiment data: {e}")
            avg_news_sentiment = 0
            avg_reddit_sentiment = 0
        
        # Analyze price trends for top cryptocurrencies
        for crypto in all_cryptos:
            try:
                # Get recent price data (last 30 days)
                thirty_days_ago = datetime.now() - timedelta(days=30)
                price_data = list(self.db["detailed-crypto-data"].find(
                    {
                        "cryptoCurrency": crypto,
                        "date": {"$gte": thirty_days_ago}
                    }
                ).sort("date", 1))
                
                if len(price_data) >= 2:
                    # Calculate price change percentage
                    first_price = price_data[0]["close"]
                    last_price = price_data[-1]["close"]
                    price_change_pct = (last_price - first_price) / first_price * 100
                    price_changes.append(price_change_pct)
                    
                    # Determine trend (simple moving average comparison)
                    if len(price_data) >= 10:
                        recent_prices = [p["close"] for p in price_data[-10:]]
                        ma5 = sum(recent_prices[-5:]) / 5
                        ma10 = sum(recent_prices) / 10
                        
                        if ma5 > ma10:
                            price_trends.append(1)  # Uptrend
                        elif ma5 < ma10:
                            price_trends.append(-1)  # Downtrend
                        else:
                            price_trends.append(0)  # Neutral
                    
                    # Volume change
                    if "volumefrom" in price_data[0] and "volumefrom" in price_data[-1]:
                        first_volume = price_data[0]["volumefrom"]
                        last_volume = price_data[-1]["volumefrom"]
                        
                        if first_volume > 0:
                            volume_change_pct = (last_volume - first_volume) / first_volume * 100
                            volume_changes.append(volume_change_pct)
            
            except Exception as e:
                print(f"Error analyzing {crypto}: {e}")
        
        # Calculate overall market score (1-100)
        # Components:
        # - Price trends: 30%
        # - Fear & Greed Index: 30%
        # - News sentiment: 20%
        # - Social media sentiment: 20%
        
        # Convert trend to a 0-100 scale
        if price_trends:
            trend_score = (sum(price_trends) / len(price_trends) + 1) * 50  # Convert -1,0,1 to 0,50,100
        else:
            trend_score = 50
            
        # Convert news sentiment from -1,1 to 0,100
        news_score = (avg_news_sentiment + 1) * 50
        
        # Convert social sentiment from -1,1 to 0,100
        social_score = (avg_reddit_sentiment + 1) * 50
        
        # Calculate weighted score
        market_score = (
            0.3 * trend_score + 
            0.3 * fear_greed_score + 
            0.2 * news_score + 
            0.2 * social_score
        )
        
        # Round to nearest integer
        market_score = round(market_score)
        
        # Interpret the score
        if market_score >= 80:
            market_interpretation = "Extremely Bullish"
        elif market_score >= 70:
            market_interpretation = "Very Bullish"
        elif market_score >= 60:
            market_interpretation = "Bullish"
        elif market_score >= 50:
            market_interpretation = "Slightly Bullish"
        elif market_score >= 40:
            market_interpretation = "Slightly Bearish"
        elif market_score >= 30:
            market_interpretation = "Bearish"
        elif market_score >= 20:
            market_interpretation = "Very Bearish"
        else:
            market_interpretation = "Extremely Bearish"
        
        # Create result dictionary
        overview = {
            "market_score": market_score,
            "interpretation": market_interpretation,
            "components": {
                "price_trend_score": trend_score,
                "fear_greed_score": fear_greed_score,
                "fear_greed_classification": fear_greed_classification,
                "news_sentiment_score": news_score,
                "social_sentiment_score": social_score
            },
            "metrics": {
                "avg_price_change_pct": sum(price_changes) / max(1, len(price_changes)),
                "avg_volume_change_pct": sum(volume_changes) / max(1, len(volume_changes))
            },
            "timestamp": datetime.now()
        }
        
        return overview
    
    def generate_market_overview_chart(self, overview: Dict, save_path: str = "market_overview.png"):
        """
        Generate a visual chart for the market overview
        
        Args:
            overview: Market overview dictionary
            save_path: Path to save the chart
        """
        # Set up the figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Crypto Market Overview: {overview['interpretation']} ({overview['market_score']}/100)", 
                    fontsize=16, fontweight='bold')
        
        # Create a gauge chart for the market score
        ax_gauge = axs[0, 0]
        score = overview['market_score']
        
        # Define colors for the gauge
        if score >= 80:
            color = 'darkgreen'
        elif score >= 60:
            color = 'limegreen'
        elif score >= 40:
            color = 'gold'
        elif score >= 20:
            color = 'orange'
        else:
            color = 'red'
            
        # Draw the gauge
        ax_gauge.set_title("Market Sentiment Score", fontsize=14)
        ax_gauge.add_patch(plt.Circle((0.5, 0), 0.4, fill=False, linewidth=2))
        ax_gauge.add_patch(plt.Rectangle((0.1, 0), 0.8, 0.1, color='lightgrey'))
        ax_gauge.add_patch(plt.Rectangle((0.1, 0), 0.8 * score/100, 0.1, color=color))
        
        # Add the score text
        ax_gauge.text(0.5, -0.15, f"{score}/100", ha='center', va='center', fontsize=18, fontweight='bold')
        ax_gauge.text(0.5, -0.25, overview['interpretation'], ha='center', va='center', fontsize=14)
        
        # Remove axes
        ax_gauge.set_xlim(0, 1)
        ax_gauge.set_ylim(-0.5, 0.5)
        ax_gauge.axis('off')
        
        # Create bar chart for component scores
        ax_components = axs[0, 1]
        components = overview['components']
        labels = ['Price Trends', 'Fear & Greed', 'News Sentiment', 'Social Sentiment']
        values = [
            components['price_trend_score'],
            components['fear_greed_score'],
            components['news_sentiment_score'],
            components['social_sentiment_score']
        ]
        
        colors = []
        for v in values:
            if v >= 80:
                colors.append('darkgreen')
            elif v >= 60:
                colors.append('limegreen')
            elif v >= 40:
                colors.append('gold')
            elif v >= 20:
                colors.append('orange')
            else:
                colors.append('red')
        
        ax_components.bar(labels, values, color=colors)
        ax_components.set_title("Component Scores", fontsize=14)
        ax_components.set_ylim(0, 100)
        ax_components.set_ylabel("Score (0-100)")
        ax_components.grid(axis='y', linestyle='--', alpha=0.7)
        plt.setp(ax_components.get_xticklabels(), rotation=45)
        
        # Create text box with key metrics
        ax_metrics = axs[1, 0]
        ax_metrics.axis('off')
        metrics_text = (
            f"Fear & Greed Index: {components['fear_greed_score']} ({components['fear_greed_classification']})\n\n"
            f"Average Price Change: {overview['metrics']['avg_price_change_pct']:.2f}%\n\n"
            f"Average Volume Change: {overview['metrics']['avg_volume_change_pct']:.2f}%\n\n"
            f"Generated on: {overview['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
        )
        ax_metrics.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_metrics.set_title("Key Metrics", fontsize=14)
        
        # Create interpretation guide
        ax_guide = axs[1, 1]
        ax_guide.axis('off')
        guide_text = (
            "Interpretation Guide:\n\n"
            "80-100: Extremely Bullish - Strong buying opportunity\n"
            "70-79: Very Bullish - Favorable market conditions\n"
            "60-69: Bullish - Positive outlook\n"
            "50-59: Slightly Bullish - Cautiously optimistic\n"
            "40-49: Slightly Bearish - Some caution advised\n"
            "30-39: Bearish - Negative outlook\n"
            "20-29: Very Bearish - Unfavorable conditions\n"
            "0-19: Extremely Bearish - High risk environment"
        )
        ax_guide.text(0.5, 0.5, guide_text, ha='center', va='center', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_guide.set_title("Market Score Interpretation", fontsize=14)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Market overview chart saved as {save_path}")
        
        return

if __name__ == "__main__":
    # Initialize the predictor
    predictor = CryptoPricePredictor(db_connection_string="mongodb://localhost:27017/", db_name="ASM")
    
    try:
        # Check command line arguments for operation mode
        import sys
        
        # Default operation: predict BTC only
        operation = "single" if len(sys.argv) <= 1 else sys.argv[1].lower()
        
        if operation == "all":
            # Predict all cryptocurrencies
            print("Starting batch prediction for all cryptocurrencies...")
            all_predictions = predictor.batch_predict(days_ahead=7)
            
            # Print summary
            print("\nPrediction Summary:")
            for crypto, pred_df in all_predictions.items():
                latest_price = pred_df.iloc[0]['predicted_price']
                last_price = pred_df.iloc[-1]['predicted_price']
                change = (last_price - latest_price) / latest_price * 100
                direction = "🔺" if change > 0 else "🔻"
                print(f"{crypto}: {direction} {abs(change):.2f}% in 7 days (Last: {last_price:.4f})")
                
        elif operation == "overview":
            # Generate market overview
            print("Generating market overview...")
            overview = predictor.get_market_overview()
            
            print("\nMarket Overview")
            print("=" * 50)
            print(f"Market Score: {overview['market_score']}/100 - {overview['interpretation']}")
            print(f"Fear & Greed Index: {overview['components']['fear_greed_score']} ({overview['components']['fear_greed_classification']})")
            print(f"Average Price Change: {overview['metrics']['avg_price_change_pct']:.2f}%")
            print(f"Average Volume Change: {overview['metrics']['avg_volume_change_pct']:.2f}%")
            
            # Generate and save the chart
            predictor.generate_market_overview_chart(overview)
            
        else:
            # Default: single cryptocurrency prediction (BTC)
            crypto_symbol = "BTC" if len(sys.argv) <= 2 else sys.argv[2].upper()
            
            print(f"Loading data for {crypto_symbol}...")
            data = predictor.load_data(crypto_symbol=crypto_symbol)
            
            # Preprocess data
            print("Preprocessing data...")
            processed_df = predictor.preprocess_data(data)
            print(f"Processed data shape: {processed_df.shape}")
            
            # Train models and evaluate (1-day forecast)
            print("Training and evaluating models...")
            metrics = predictor.evaluate_model(processed_df, forecast_horizon=1, plot=True)
            print(f"Best model: {predictor.best_model}")
            print(f"Metrics: {metrics[predictor.best_model]}")
            
            # Get feature importance
            importance = predictor.feature_importance()
            print("\nTop 10 most important features:")
            print(importance.head(10))
            
            # Make predictions for next 7 days
            print("\nPredicting prices for the next 7 days...")
            future_predictions = predictor.predict_future(processed_df, days_ahead=7)
            print("Predictions:")
            print(future_predictions)
            
            print("\nModel evaluation plot saved as 'model_evaluation.png'")
            
        print("\nOperation completed successfully.")
    
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
