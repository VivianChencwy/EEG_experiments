"""Visualization utilities for EEG experiments."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

def plot_confusion_matrix(y_true, y_pred, y_proba=None, save_path=None):
    """
    Plot and save confusion matrix with metrics.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : array-like, optional
        Prediction probabilities for positive class (for AUC calculation)
    save_path : str, optional
        Path to save the plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate metrics
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    # Calculate AUC if probabilities are provided
    auc = 0.5  # default value
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
        except:
            pass  # keep default AUC if calculation fails
    
    # Create figure and axes
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add metrics text box
    metrics_text = f'Accuracy: {accuracy:.4f}\n'
    metrics_text += f'Precision: {precision:.4f}\n'
    metrics_text += f'Recall: {recall:.4f}\n'
    metrics_text += f'F1 Score: {f1:.4f}\n'
    metrics_text += f'AUC: {auc:.4f}'
    
    plt.text(2.5, 1.5, metrics_text,
             bbox=dict(facecolor='white', alpha=0.8),
             fontsize=10, ha='left', va='center')
    
    # Save plot if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }
