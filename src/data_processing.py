"""
Data processing utilities for machine learning projects.

This module contains functions for common data processing tasks
such as cleaning, transformation, and feature engineering.
"""

from typing import List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from various file formats.
    
    Args:
        file_path: Path to the data file
        **kwargs: Additional arguments for pandas read functions
        
    Returns:
        DataFrame containing the loaded data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, **kwargs)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path, **kwargs)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
            
        logger.info(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
        return df
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def clean_data(df: pd.DataFrame, 
               drop_na: bool = True,
               fill_numeric_na: Optional[Union[str, float]] = 'mean',
               fill_categorical_na: str = 'mode') -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and duplicates.
    
    Args:
        df: Input DataFrame
        drop_na: Whether to drop rows with NaN values
        fill_numeric_na: Strategy for filling numeric NaN values ('mean', 'median', or numeric value)
        fill_categorical_na: Strategy for filling categorical NaN values ('mode' or string value)
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Remove duplicates
    initial_shape = df_clean.shape
    df_clean = df_clean.drop_duplicates()
    logger.info(f"Removed {initial_shape[0] - df_clean.shape[0]} duplicate rows")
    
    if drop_na:
        df_clean = df_clean.dropna()
        logger.info(f"Dropped rows with NaN values. New shape: {df_clean.shape}")
    else:
        # Fill numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isna().any():
                if fill_numeric_na == 'mean':
                    fill_value = df_clean[col].mean()
                elif fill_numeric_na == 'median':
                    fill_value = df_clean[col].median()
                else:
                    fill_value = fill_numeric_na
                df_clean[col].fillna(fill_value, inplace=True)
        
        # Fill categorical columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isna().any():
                if fill_categorical_na == 'mode':
                    fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                else:
                    fill_value = fill_categorical_na
                df_clean[col].fillna(fill_value, inplace=True)
    
    return df_clean


def split_features_target(df: pd.DataFrame, 
                         target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target variable.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        
    Returns:
        Tuple of (features DataFrame, target Series)
        
    Raises:
        KeyError: If target_column is not in DataFrame
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in DataFrame")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    logger.info(f"Split data into features (shape: {X.shape}) and target (shape: {y.shape})")
    return X, y


def encode_categorical_features(X: pd.DataFrame, 
                              categorical_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        X: Features DataFrame
        categorical_columns: List of categorical column names. If None, auto-detect.
        
    Returns:
        Tuple of (encoded DataFrame, dictionary of encoders)
    """
    X_encoded = X.copy()
    encoders = {}
    
    if categorical_columns is None:
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_columns:
        if col in X.columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
            logger.info(f"Encoded categorical column: {col}")
    
    return X_encoded, encoders


def scale_features(X_train: pd.DataFrame, 
                  X_test: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], StandardScaler]:
    """
    Scale features using StandardScaler.
    
    Args:
        X_train: Training features
        X_test: Test features (optional)
        
    Returns:
        Tuple of (scaled X_train, scaled X_test, fitted scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
    
    logger.info("Features scaled using StandardScaler")
    return X_train_scaled, X_test_scaled, scaler


def create_train_test_split(X: pd.DataFrame, 
                          y: pd.Series,
                          test_size: float = 0.2,
                          random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create train-test split of the data.
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of test set
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.dtype == 'object' else None
    )
    
    logger.info(f"Created train-test split. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test
