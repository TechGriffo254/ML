"""
Machine learning model utilities.

This module contains functions for training, evaluating, and saving ML models.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import joblib
import logging

logger = logging.getLogger(__name__)


def train_model(model: Any, 
                X_train: pd.DataFrame, 
                y_train: pd.Series) -> Any:
    """
    Train a machine learning model.
    
    Args:
        model: Sklearn model instance
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained model
    """
    try:
        model.fit(X_train, y_train)
        logger.info(f"Successfully trained {model.__class__.__name__}")
        return model
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise


def evaluate_classification_model(model: Any,
                                X_test: pd.DataFrame,
                                y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluate a classification model.
    
    Args:
        model: Trained classification model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'predictions': y_pred
    }
    
    logger.info(f"Classification accuracy: {metrics['accuracy']:.4f}")
    return metrics


def evaluate_regression_model(model: Any,
                            X_test: pd.DataFrame,
                            y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluate a regression model.
    
    Args:
        model: Trained regression model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'predictions': y_pred
    }
    
    logger.info(f"Regression R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
    return metrics


def cross_validate_model(model: Any,
                        X: pd.DataFrame,
                        y: pd.Series,
                        cv: int = 5,
                        scoring: str = 'accuracy') -> Dict[str, float]:
    """
    Perform cross-validation on a model.
    
    Args:
        model: Model to evaluate
        X: Features
        y: Target
        cv: Number of cross-validation folds
        scoring: Scoring metric
        
    Returns:
        Dictionary with cross-validation results
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    results = {
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'scores': scores
    }
    
    logger.info(f"Cross-validation {scoring}: {results['mean_score']:.4f} (+/- {results['std_score'] * 2:.4f})")
    return results


def save_model(model: Any, 
               filepath: str,
               model_metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model
        filepath: Path to save the model
        model_metadata: Optional metadata about the model
    """
    try:
        # Save model
        joblib.dump(model, filepath)
        
        # Save metadata if provided
        if model_metadata:
            metadata_path = filepath.replace('.pkl', '_metadata.pkl')
            joblib.dump(model_metadata, metadata_path)
            
        logger.info(f"Model saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise


def load_model(filepath: str) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Load a saved model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Tuple of (model, metadata)
    """
    try:
        model = joblib.load(filepath)
        
        # Try to load metadata
        metadata = None
        metadata_path = filepath.replace('.pkl', '_metadata.pkl')
        try:
            metadata = joblib.load(metadata_path)
        except FileNotFoundError:
            pass
            
        logger.info(f"Model loaded from {filepath}")
        return model, metadata
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def get_feature_importance(model: Any, 
                          feature_names: list) -> pd.DataFrame:
    """
    Get feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance scores
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df
