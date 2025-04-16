import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from lightgbm import LGBMRegressor
import joblib
import os
from datetime import datetime

def analyze_feature_importance(model, features_list, top_n=15, plot=True, save_path=None):
    """
    Analyze and visualize feature importance from the model
    
    Args:
        model: Trained model (must have feature_importances_ attribute)
        features_list: List of feature names
        top_n: Number of top features to display
        plot: Whether to generate a plot
        save_path: Path to save the feature importance plot
        
    Returns:
        DataFrame with feature importances
    """
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame({
        'Feature': features_list,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Plot feature importances if requested
    if plot:
        # Get top N features
        top_features = feature_importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(top_features['Feature'][::-1], top_features['Importance'][::-1])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    return feature_importance_df

def optimize_random_forest(X_train, y_train, cv=5):
    """
    Optimize Random Forest hyperparameters using grid search
    
    Args:
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        
    Returns:
        Best model and parameters
    """
    print("Optimizing Random Forest model...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Use TimeSeriesSplit for time series data
    tscv = TimeSeriesSplit(n_splits=cv)
    
    # Create base model
    rf = RandomForestRegressor(random_state=42)
    
    # Setup grid search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    print(f"Best Random Forest parameters: {grid_search.best_params_}")
    print(f"Best score: {-grid_search.best_score_:.4f} (MSE)")
    
    return grid_search.best_estimator_, grid_search.best_params_

def optimize_xgboost(X_train, y_train, cv=5):
    """
    Optimize XGBoost hyperparameters using grid search
    
    Args:
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        
    Returns:
        Best model and parameters
    """
    print("Optimizing XGBoost model...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Use TimeSeriesSplit for time series data
    tscv = TimeSeriesSplit(n_splits=cv)
    
    # Create base model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Setup grid search
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    print(f"Best XGBoost parameters: {grid_search.best_params_}")
    print(f"Best score: {-grid_search.best_score_:.4f} (MSE)")
    
    return grid_search.best_estimator_, grid_search.best_params_

def optimize_lightgbm(X_train, y_train, cv=5):
    """
    Optimize LightGBM hyperparameters using grid search
    
    Args:
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        
    Returns:
        Best model and parameters
    """
    print("Optimizing LightGBM model...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'num_leaves': [31, 63, 127],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Use TimeSeriesSplit for time series data
    tscv = TimeSeriesSplit(n_splits=cv)
    
    # Create base model
    lgbm_model = LGBMRegressor(random_state=42)
    
    # Setup grid search
    grid_search = GridSearchCV(
        estimator=lgbm_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    print(f"Best LightGBM parameters: {grid_search.best_params_}")
    print(f"Best score: {-grid_search.best_score_:.4f} (MSE)")
    
    return grid_search.best_estimator_, grid_search.best_params_

def train_ensemble(models, X_train, y_train, X_test, y_test):
    """
    Train an ensemble of models and evaluate their performance
    
    Args:
        models: Dictionary of model name -> model object
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary of evaluation results
    """
    print("Training ensemble of models...")
    
    # Dictionary to store predictions and metrics
    predictions = {}
    metrics = {}
    
    # Train each model and make predictions
    for name, model in models.items():
        print(f"Training {name} model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        pred = model.predict(X_test)
        predictions[name] = pred
        
        # Calculate metrics
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
        
        metrics[name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        print(f"{name} metrics:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
    
    # Create ensemble prediction (simple average)
    all_preds = np.column_stack([predictions[name] for name in predictions.keys()])
    ensemble_pred = np.mean(all_preds, axis=1)
    
    # Calculate ensemble metrics
    mse = mean_squared_error(y_test, ensemble_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, ensemble_pred)
    r2 = r2_score(y_test, ensemble_pred)
    mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
    
    metrics['ensemble'] = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }
    
    print("Ensemble metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {
        'predictions': predictions,
        'ensemble_pred': ensemble_pred,
        'metrics': metrics
    }

def plot_model_comparison(results, y_test, save_path=None):
    """
    Plot a comparison of model predictions versus actual values
    
    Args:
        results: Dictionary with model predictions and metrics
        y_test: Actual test values
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot actual values
    plt.plot(y_test.index, y_test, label='Actual', color='black', linewidth=2)
    
    # Plot predictions for each model
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for i, (name, pred) in enumerate(results['predictions'].items()):
        plt.plot(y_test.index, pred, label=f'{name} (RMSE: {results["metrics"][name]["rmse"]:.2f})', 
                 color=colors[i % len(colors)], linestyle='--')
    
    # Plot ensemble prediction
    plt.plot(y_test.index, results['ensemble_pred'], label=f'Ensemble (RMSE: {results["metrics"]["ensemble"]["rmse"]:.2f})', 
             color='magenta', linewidth=2, linestyle='-.')
    
    plt.title('Model Comparison: Predicted vs Actual Values')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Model comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def save_models(models, model_dir='model_exports'):
    """
    Save trained models to disk
    
    Args:
        models: Dictionary of model name -> model object
        model_dir: Directory to save models
        
    Returns:
        Dictionary with paths to saved models
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_paths = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for name, model in models.items():
        path = f"{model_dir}/{name}_{timestamp}.joblib"
        joblib.dump(model, path)
        model_paths[name] = path
        print(f"Saved {name} model to {path}")
    
    return model_paths
