"""
Tests for data processing utilities.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_processing import (
    load_data, clean_data, split_features_target,
    encode_categorical_features, scale_features, create_train_test_split
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'numeric_col1': [1, 2, 3, 4, 5],
        'numeric_col2': [10, 20, 30, 40, 50],
        'categorical_col': ['A', 'B', 'A', 'C', 'B'],
        'target': [0, 1, 0, 1, 0]
    })


@pytest.fixture
def sample_dataframe_with_na():
    """Create a sample DataFrame with missing values for testing."""
    return pd.DataFrame({
        'numeric_col1': [1, 2, np.nan, 4, 5],
        'numeric_col2': [10, np.nan, 30, 40, 50],
        'categorical_col': ['A', 'B', None, 'C', 'B'],
        'target': [0, 1, 0, 1, 0]
    })


def test_split_features_target(sample_dataframe):
    """Test splitting features and target."""
    X, y = split_features_target(sample_dataframe, 'target')
    
    assert X.shape == (5, 3)
    assert y.shape == (5,)
    assert 'target' not in X.columns
    assert y.name == 'target'


def test_split_features_target_invalid_column(sample_dataframe):
    """Test splitting with invalid target column."""
    with pytest.raises(KeyError):
        split_features_target(sample_dataframe, 'invalid_column')


def test_clean_data_drop_na(sample_dataframe_with_na):
    """Test cleaning data by dropping NaN values."""
    df_clean = clean_data(sample_dataframe_with_na, drop_na=True)
    
    assert df_clean.isna().sum().sum() == 0
    assert df_clean.shape[0] < sample_dataframe_with_na.shape[0]


def test_clean_data_fill_na(sample_dataframe_with_na):
    """Test cleaning data by filling NaN values."""
    df_clean = clean_data(sample_dataframe_with_na, drop_na=False)
    
    assert df_clean.isna().sum().sum() == 0
    assert df_clean.shape[0] == sample_dataframe_with_na.shape[0]


def test_encode_categorical_features(sample_dataframe):
    """Test encoding categorical features."""
    X, _ = split_features_target(sample_dataframe, 'target')
    X_encoded, encoders = encode_categorical_features(X)
    
    assert 'categorical_col' in encoders
    assert X_encoded['categorical_col'].dtype in [np.int32, np.int64]


def test_scale_features(sample_dataframe):
    """Test feature scaling."""
    X, _ = split_features_target(sample_dataframe, 'target')
    X_numeric = X.select_dtypes(include=[np.number])
    
    X_scaled, _, scaler = scale_features(X_numeric)
    
    # Check that mean is approximately 0 and std is approximately 1
    assert np.allclose(X_scaled.mean(), 0, atol=1e-10)
    assert np.allclose(X_scaled.std(), 1, atol=1e-10)


def test_create_train_test_split(sample_dataframe):
    """Test train-test split creation."""
    X, y = split_features_target(sample_dataframe, 'target')
    X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.4)
    
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0]
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
