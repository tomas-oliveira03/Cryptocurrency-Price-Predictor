import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from bson import CodecOptions
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
        
        # Capture X_test and y_test
        y_test_initial = model_results['y_test']

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
            lstm_results = None

        # Plot Comparison of All Models on Test Set
        print("\nGenerating comparison plot for all models on the test set...")
        predictions_for_plot = {}
        metrics_for_plot = {}
        all_preds_list = []

        # Add predictions from initial models (RF, XGB, MA)
        for model_name, results in model_results['all_metrics'].items():
            predictions_for_plot[model_name] = results['pred']
            metrics_for_plot[model_name] = {k: v for k, v in results.items() if k in ['mse', 'rmse', 'mae', 'r2', 'mape']}
            if model_name != 'moving_avg':  # Don't include MA in ensemble average
                all_preds_list.append(results['pred'])

        # Add LSTM predictions if available and successful
        if use_lstm and lstm_results:
            # Align LSTM predictions with the y_test_initial index
            lstm_test_preds_aligned = lstm_results['results']['predicted'].reindex(y_test_initial.index)
            # --- Check for NaNs after reindex/fill ---
            filled_lstm_preds = lstm_test_preds_aligned.bfill().ffill()
            if filled_lstm_preds.isnull().any():
                 print("Warning: NaNs still present in LSTM predictions after filling. Plotting might be affected.")
            # -----------------------------------------
            predictions_for_plot['lstm'] = filled_lstm_preds
            metrics_for_plot['lstm'] = lstm_results['metrics']
            # --- Add only if not all NaN ---
            if not filled_lstm_preds.isnull().all():
                 all_preds_list.append(filled_lstm_preds)
            else:
                 print("Warning: LSTM predictions are all NaN after alignment.")
            # -------------------------------

        # Calculate a simple ensemble average for plotting, ignoring NaNs
        if all_preds_list:
            # Ensure all arrays are numpy arrays for averaging
            valid_preds = []
            for pred in all_preds_list:
                 if isinstance(pred, (np.ndarray, pd.Series)) and len(pred) == len(y_test_initial):
                      # --- Check for all NaNs before adding ---
                      if isinstance(pred, pd.Series) and pred.isnull().all():
                           print(f"Warning: Skipping an all-NaN prediction series during ensemble calculation.")
                           continue
                      elif isinstance(pred, np.ndarray) and np.isnan(pred).all():
                           print(f"Warning: Skipping an all-NaN prediction array during ensemble calculation.")
                           continue
                      # --------------------------------------
                      valid_preds.append(pred.values if isinstance(pred, pd.Series) else pred)

            if valid_preds:
                # --- Use np.nanmean to calculate average, ignoring NaNs ---
                ensemble_pred_for_plot = np.nanmean(np.array(valid_preds), axis=0)
                # ----------------------------------------------------------
            else:
                print("Warning: No valid predictions found for ensemble calculation.")
                ensemble_pred_for_plot = np.full(len(y_test_initial), np.nan) # Fill with NaN if no valid preds
        else:
            print("Warning: No base predictions available for ensemble calculation.")
            ensemble_pred_for_plot = np.full(len(y_test_initial), np.nan) # Fill with NaN

        # --- Add Debug Print ---
        print(f"Ensemble prediction contains NaNs: {np.isnan(ensemble_pred_for_plot).any()}")
        # -----------------------

        # Calculate metrics for the ensemble prediction
        # Check for NaNs before calculating metrics (use np.isfinite on the potentially NaN result from nanmean)
        valid_ensemble_idx = np.isfinite(y_test_initial.values) & np.isfinite(ensemble_pred_for_plot)
        if valid_ensemble_idx.sum() > 0: # Check if there are any valid points to compare
            y_true_valid = y_test_initial.values[valid_ensemble_idx]
            y_pred_valid = ensemble_pred_for_plot[valid_ensemble_idx]

            ensemble_mse = mean_squared_error(y_true_valid, y_pred_valid)
            ensemble_rmse = np.sqrt(ensemble_mse)
            ensemble_mae = mean_absolute_error(y_true_valid, y_pred_valid)
            ensemble_r2 = r2_score(y_true_valid, y_pred_valid)

            # Handle potential division by zero in MAPE for valid points
            valid_mape_indices = y_true_valid != 0
            if valid_mape_indices.any():
                 ensemble_mape = np.mean(np.abs((y_true_valid[valid_mape_indices] - y_pred_valid[valid_mape_indices]) / y_true_valid[valid_mape_indices])) * 100
            else:
                 ensemble_mape = np.nan

            metrics_for_plot['ensemble'] = {
                'mse': ensemble_mse, 'rmse': ensemble_rmse, 'mae': ensemble_mae,
                'r2': ensemble_r2, 'mape': ensemble_mape
            }
            print(f"Calculated ensemble metrics (RMSE: {ensemble_rmse:.2f}) using {valid_ensemble_idx.sum()} valid points.")
        else:
            print("Warning: No valid overlapping points found between y_test and ensemble predictions. Skipping ensemble metrics calculation for plot.")
            metrics_for_plot['ensemble'] = {'rmse': np.nan, 'mse': np.nan, 'mae': np.nan, 'r2': np.nan, 'mape': np.nan} # Ensure all keys exist

        comparison_results = {
            'predictions': predictions_for_plot,
            'metrics': metrics_for_plot,
            'ensemble_pred': ensemble_pred_for_plot
        }

        # --- Add Debug Print ---
        print(f"Models included in plot data: {list(comparison_results['predictions'].keys())}")
        # -----------------------

        modelOptimizer.plot_model_comparison(
            comparison_results,
            y_test_initial,
            save_path="Services/Models/data/all_models_test_comparison.png"
        )

        # Select best model for predictions (based on initial comparison or LSTM)
        if use_lstm and lstm_results:
            best_model_for_future = lstm_results['model']  # Use LSTM if it's better
            print("\nSelected LSTM model for future predictions.")
            # --- Use LSTM backtesting function ---
            print("\nRunning backtest on historical data (LSTM)...")
            backtest_results = lstmModel.backtest_lstm_model(
                lstm_results,
                featuresDF
            )
            # ------------------------------------
        else:
            best_model_for_future = model_results['model']  # Use best from initial training
            print(f"\nSelected {model_results['best_model']} model for future predictions.")
            # --- Use simple model backtesting function ---
            print("\nRunning backtest on historical data (Tree-based)...")
            backtest_results = simplePredictionModel.backtest_model(
                best_model_for_future,
                model_results['scaler'],
                model_results['features'],
                featuresDF
            )
            # -----------------------------------------

        # Make predictions for the next 5 days
        print("\nMaking predictions for the next 5 days...")
        # --- Use the correct prediction function based on the selected model ---
        if use_lstm and lstm_results:
             future_predictions = lstmModel.predict_with_lstm(
                  lstm_results,
                  featuresDF,
                  days=5
             )
        else:
             future_predictions = simplePredictionModel.make_future_predictions(
                  best_model_for_future,
                  model_results['scaler'],
                  model_results['features'],
                  featuresDF,
                  days=5
             )
        # --------------------------------------------------------------------
        
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
        # --- Ensure feature importance uses a tree-based model ---
        # Use the best model from the initial simple training run for importance
        initial_best_tree_model_name = model_results['best_model']
        if initial_best_tree_model_name != 'moving_avg':
             initial_best_tree_model = model_results['all_metrics'][initial_best_tree_model_name]['model']
             feature_importance = modelOptimizer.analyze_feature_importance(
                  initial_best_tree_model,
                  model_results['features'], # Use features from initial run
                  save_path="Services/Models/data/feature_importance.png"
             )
             print("\nTop 10 most important features:")
             print(feature_importance.head(10))
        else:
             print("Skipping feature importance analysis (best initial model was Moving Average).")
             feature_importance = pd.DataFrame() # Empty dataframe
        # -------------------------------------------------------
        
        # For LSTM path - Future predictions already handled above
        if use_lstm:
            pass # Future predictions handled before detailed printout
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
                    
                    # Make predictions using the best optimized model
                    print("\nMaking predictions with the best optimized model...")
                    # --- Update future_predictions if optimization occurred and improved ---
                    if best_model_for_future != model_results['model']: # Check if optimization changed the model
                         future_predictions = simplePredictionModel.make_future_predictions(
                              best_model_for_future, # Use the potentially updated best model
                              model_results['scaler'],
                              model_results['features'],
                              featuresDF,
                              days=5
                         )
                    # -----------------------------------------------------------------------
                else: # If ensemble wasn't trained (e.g., only one optimized model)
                    print("Only one model available after optimization or optimization failed; using initial best model predictions.")
                    pass # Assuming future_predictions from initial best model is sufficient if optimization is limited

        # Step 7: Print and visualize FINAL future predictions
        print("\nFinal Future Price Predictions:")
        for date, row in future_predictions.iterrows():
            print(f"  {date.strftime('%Y-%m-%d')}: ${row['predicted_price']:.2f}")
        
        print("\nCreating visualization...")
        # Pass the relevant sentiment column if it exists in featuresDF
        sentiment_col = 'pct_positive' if 'pct_positive' in featuresDF.columns else None
        sentiment_plot_data = featuresDF[[sentiment_col]] if sentiment_col else None
        
        simplePredictionModel.visualize_predictions(
            featuresDF[['close']],
            future_predictions,
            sentiment_data=sentiment_plot_data, # Pass sentiment data here
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
