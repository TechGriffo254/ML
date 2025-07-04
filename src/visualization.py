"""
Visualization utilities for machine learning projects.

This module contains functions for creating various plots and visualizations
commonly used in data science and ML workflows.
"""

from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


def plot_data_distribution(df: pd.DataFrame, 
                          columns: Optional[List[str]] = None,
                          figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot distribution of numerical columns in the dataset.
    
    Args:
        df: Input DataFrame
        columns: List of columns to plot. If None, plot all numerical columns
        figsize: Figure size tuple
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(columns):
        if i < len(axes):
            df[col].hist(bins=30, ax=axes[i], alpha=0.7)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide unused subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, 
                           figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Plot correlation matrix heatmap.
    
    Args:
        df: Input DataFrame
        figsize: Figure size tuple
    """
    # Select only numerical columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        logger.warning("No numerical columns found for correlation matrix")
        return
    
    correlation_matrix = numeric_df.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size tuple
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importance_df: pd.DataFrame,
                           top_n: int = 15,
                           figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
        figsize: Figure size tuple
    """
    if importance_df.empty:
        logger.warning("Feature importance DataFrame is empty")
        return
    
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=figsize)
    sns.barplot(data=top_features, y='feature', x='importance')
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()


def plot_learning_curve(model, X, y, 
                       cv: int = 5,
                       train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
                       figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot learning curves for a model.
    
    Args:
        model: ML model
        X: Features
        y: Target
        cv: Cross-validation folds
        train_sizes: Training set sizes
        figsize: Figure size tuple
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes, 
        scoring='accuracy' if hasattr(y, 'nunique') and y.nunique() < 20 else 'r2'
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, 
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, 
                     alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, 
                     val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true: np.ndarray, 
                  y_prob: np.ndarray,
                  figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        figsize: Figure size tuple
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true: np.ndarray, 
                  y_pred: np.ndarray,
                  figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot residuals for regression models.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        figsize: Figure size tuple
    """
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted')
    ax1.grid(True)
    
    # Histogram of residuals
    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Residuals')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
