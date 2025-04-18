import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from bson import CodecOptions
from pymongo import MongoClient
from sklearn.model_selection import train_test_split

import fetchData
import preProcessor
import engineFeatures
import simplePredictionModel
import modelOptimizer
import lstmModel  # Import the new LSTM module


class PredictionModel:
    def __init__(self):
        
        mongoDBURI = os.getenv("MONGODB_URI")
        if not mongoDBURI:
            raise ValueError("Please set the MONGODB_URI environment variable first.")

        # Database connection
        mongoClient = MongoClient(mongoDBURI)
        
        self.cryptoFearGreedDB = mongoClient['ASM'].get_collection('crypto-fear-greed', codec_options=CodecOptions(tz_aware=True))
        self.detailedCryptoData = mongoClient['ASM'].get_collection('detailed-crypto-data', codec_options=CodecOptions(tz_aware=True))
        
        self.redditDB = mongoClient['ASM'].get_collection('reddit', codec_options=CodecOptions(tz_aware=True))
        self.forumDB = mongoClient['ASM'].get_collection('forum', codec_options=CodecOptions(tz_aware=True))
        self.articlesDB = mongoClient['ASM'].get_collection('articles', codec_options=CodecOptions(tz_aware=True))
        

    def runEverything(self):
        # Step 1: Get raw data for prediction
        rawData = fetchData.getDataForPrediction(self, cryptoCoin="BTC", numberOfPastDaysOfData=365)
        
        # Step 2: Preprocess the data
        processedData = preProcessor.preprocessData(self, rawData)
        
        # Step 3: Engineer features
        featuresDF = engineFeatures.engineFeatures(self, processedData)
        
        # Print feature summary
        print("\nEngineered Features Summary:")
        print(f"Total features: {len(featuresDF.columns)}")
        print(f"Total data points: {len(featuresDF)}")
        print("Feature names:", ", ".join(featuresDF.columns.tolist()[:5]) + "...")
        
        # Save raw feature data first
        csv_dir = "Services/Models/data/data_exports"
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        
        features_csv = f"{csv_dir}/raw_features.csv"
        featuresDF.to_csv(features_csv)
        print(f"Raw features saved to {features_csv}")
        
        # Step 4: Train standard prediction model and evaluate
        print("Training prediction models...")
        model_results = simplePredictionModel.train_model(
            featuresDF, 
            target_column='close', 
            forecast_days=5
        )
        
        # Print model evaluation metrics
        print("\nBest Model Evaluation Metrics:")
        print(f"Best model: {model_results.get('best_model', 'unknown')}")
        for metric, value in model_results['metrics'].items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        # Print comparison of all models
        print("\nAll Models Comparison:")
        for model_name, metrics in model_results.get('all_metrics', {}).items():
            print(f"  {model_name.upper()}:")
            for metric in ['rmse', 'mape', 'r2']:
                if metric in metrics:
                    print(f"    {metric.upper()}: {metrics[metric]:.4f}")
        
        # Step 5: Train LSTM model
        try:
            print("\nTraining LSTM model...")
            # Use a smaller sequence length if dataset is smaller
            seq_length = min(10, len(featuresDF) // 5)  # Adjust sequence length based on data size
            
            lstm_results = lstmModel.train_lstm_model(
                featuresDF,
                target_column='close',
                forecast_days=5,
                test_size=0.2,
                seq_length=seq_length
            )
            
            print("\nLSTM Model Evaluation Metrics:")
            for metric, value in lstm_results['metrics'].items():
                print(f"  {metric.upper()}: {value:.4f}")
                
            # Decide whether to use LSTM based on performance comparison
            if lstm_results['metrics']['rmse'] < model_results['metrics']['rmse']:
                print("\nLSTM model performs better - using it for predictions")
                use_lstm = True
            else:
                print("\nTree-based model performs better - continuing with optimization")
                use_lstm = False
                
        except Exception as e:
            print(f"\nError training LSTM model: {e}")
            print("Continuing with tree-based models only")
            use_lstm = False
        
        # Select best model for predictions
        best_model = model_results['model']
        
        # Run backtest to evaluate model performance on historical data
        print("\nRunning backtest on historical data...")
        backtest_results = simplePredictionModel.backtest_model(
            best_model,
            model_results['scaler'],
            model_results['features'],
            featuresDF
        )
        
        # Make predictions for the next 5 days
        print("\nMaking predictions for the next 5 days...")
        future_predictions = simplePredictionModel.make_future_predictions(
            best_model,
            model_results['scaler'],
            model_results['features'],
            featuresDF,
            days=5
        )
        
        # Print detailed prediction information
        print("\nDetailed Price Predictions:")
        for date, row in future_predictions.iterrows():
            # Calculate day-over-day change if not the first prediction
            if date == future_predictions.index[0]:
                last_price = featuresDF['close'].iloc[-1]
                day_change_pct = ((row['predicted_price'] - last_price) / last_price) * 100
                print(f"  {date.strftime('%Y-%m-%d')}: ${row['predicted_price']:.2f} ({day_change_pct:+.2f}% from last known price)")
            else:
                # Use iloc instead of direct position-based indexing
                idx = future_predictions.index.get_loc(date) - 1
                prev_price = future_predictions['predicted_price'].iloc[idx]
                day_change_pct = ((row['predicted_price'] - prev_price) / prev_price) * 100
                print(f"  {date.strftime('%Y-%m-%d')}: ${row['predicted_price']:.2f} ({day_change_pct:+.2f}% daily change)")
        
        # Calculate overall prediction trend
        first_pred = future_predictions['predicted_price'].iloc[0]
        last_pred = future_predictions['predicted_price'].iloc[-1]
        total_change_pct = ((last_pred - first_pred) / first_pred) * 100
        print(f"\nOverall {len(future_predictions)}-day prediction trend: {total_change_pct:+.2f}%")
        
        # Step 6: Optimize tree-based models with hyperparameter tuning
        print("\nStarting model optimization...")
        
        # Create train/test split
        X = featuresDF.drop('target', axis=1) if 'target' in featuresDF.columns else featuresDF
        y = featuresDF['close']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Analyze feature importance of the basic model
        print("\nAnalyzing feature importance...")
        feature_importance = modelOptimizer.analyze_feature_importance(
            model_results['model'],
            model_results['features'],
            save_path="Services/Models/data/feature_importance.png"
        )
        
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
        
        # For LSTM path
        if use_lstm:
            print("\nMaking predictions with LSTM model...")
            future_predictions = lstmModel.predict_with_lstm(
                lstm_results,
                featuresDF,
                days=5
            )
        # For tree-based models path
        else:
            # Optimize models if dataset is large enough (skip for small datasets)
            if len(X_train) > 50:
                print("\nOptimizing models with hyperparameter tuning...")
                
                # Create a set of optimized models
                optimized_models = {}
                
                # Optimize Random Forest
                try:
                    rf_model, rf_params = modelOptimizer.optimize_random_forest(X_train, y_train, cv=5)
                    optimized_models['random_forest'] = rf_model
                except Exception as e:
                    print(f"Error optimizing Random Forest: {e}")
                
                # Optimize XGBoost
                try:
                    xgb_model, xgb_params = modelOptimizer.optimize_xgboost(X_train, y_train, cv=5)
                    optimized_models['xgboost'] = xgb_model
                except Exception as e:
                    print(f"Error optimizing XGBoost: {e}")
                
                # Add basic model as fallback
                optimized_models['basic_rf'] = model_results['model']
                
                # Train ensemble and evaluate
                if len(optimized_models) > 1:
                    print("\nTraining ensemble model...")
                    ensemble_results = modelOptimizer.train_ensemble(
                        optimized_models, X_train, y_train, X_test, y_test
                    )
                    
                    # Plot model comparison
                    modelOptimizer.plot_model_comparison(
                        ensemble_results, y_test, save_path="Services/Models/data/model_comparison.png"
                    )
                    
                    # Save the best models
                    model_paths = modelOptimizer.save_models(optimized_models)
                    print(f"\nSaved {len(model_paths)} models to disk.")
                    
                    # Use the best model for predictions
                    best_model_name = min(ensemble_results['metrics'], 
                                         key=lambda x: ensemble_results['metrics'][x]['rmse'])
                    
                    print(f"\nBest model: {best_model_name}")
                    best_model = optimized_models.get(best_model_name, model_results['model'])
                    
                    # Make predictions using the best model
                    print("\nMaking predictions with the best model...")
                    future_predictions = simplePredictionModel.make_future_predictions(
                        best_model,
                        model_results['scaler'],
                        model_results['features'],
                        featuresDF,
                        days=5
                    )
        
        # Step 7: Print and visualize predictions
        print("\nPrice Predictions:")
        for date, row in future_predictions.iterrows():
            print(f"  {date.strftime('%Y-%m-%d')}: ${row['predicted_price']:.2f}")
        
        print("\nCreating visualization...")
        simplePredictionModel.visualize_predictions(
            featuresDF[['close']],
            future_predictions,
            save_path="Services/Models/data/crypto_price_prediction.png"
        )
        
        # Save the engineered features to CSV
        csv_dir = "Services/Models/data/data_exports"
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        
        # Save predictions to CSV
        predictions_csv = f"{csv_dir}/price_predictions.csv"
        future_predictions.to_csv(predictions_csv)
        print(f"Predictions saved to {predictions_csv}")
        
        # Return results dictionary
        results_dict = {
            "features": featuresDF,
            "model_results": model_results,
            "predictions": future_predictions,
            "feature_importance": feature_importance.head(20).to_dict()
        }
        
        # Add LSTM results if available
        if use_lstm:
            results_dict["lstm_results"] = lstm_results
            
        return results_dict

if __name__ == "__main__":
    # Example usage
    predictionModel = PredictionModel()
    results = predictionModel.runEverything()
