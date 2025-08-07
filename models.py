"""
Model definitions and related functions for EEG experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from braindecode.models import ShallowFBCSPNet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from config import (
    INPUT_WINDOW_SAMPLES, use_subject_layer, EARLY_STOPPING_PATIENCE,
    LEARNING_RATE, WEIGHT_DECAY, GAMMA, MAX_EPOCHS, N_CLASSES
)
from constants import NORMALIZATION_EPSILON


class SubjectInputLayer(nn.Module):
    """Subject-specific input layer for personalized EEG processing.
    
    Each subject gets their own linear transformation matrix applied to the input.
    """
    def __init__(self, n_subjects, n_channels):
        super().__init__()
        # Initialize with identity matrices (no transformation initially)
        self.weights = nn.Parameter(torch.eye(n_channels).unsqueeze(0).repeat(n_subjects, 1, 1))
        self.n_subjects = n_subjects
        self.n_channels = n_channels
    
    def forward(self, x, subject_indices):
        """
        Apply subject-specific linear transformation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input EEG data, shape (batch_size, n_channels, n_timepoints)
        subject_indices : torch.Tensor
            Subject indices for each sample, shape (batch_size,)
            
        Returns
        -------
        torch.Tensor
            Transformed EEG data, same shape as input
        """
        batch_size = x.size(0)
        # Get subject-specific weights: (batch_size, n_channels, n_channels)
        subject_weights = self.weights[subject_indices]  
        # Apply transformation: (batch_size, n_channels, n_timepoints)
        return torch.einsum('bct,bcd->bdt', x, subject_weights)


class ShallowFBCSPNetWithSubjectLayer(nn.Module):
    """Wrapper that adds subject layer to ShallowFBCSPNet."""
    def __init__(self, subject_layer, base_model):
        super().__init__()
        self.subject_layer = subject_layer
        self.base_model = base_model
    
    def forward(self, x, subject_indices=None):
        if subject_indices is not None:
            x = self.subject_layer(x, subject_indices)
        return self.base_model(x)


def create_model(n_channels, is_lda=False, random_state=None, n_subjects=None, enable_subject_layer=None):
    """Create a new model based on configuration.
    
    Parameters
    ----------
    n_channels : int
        Number of input channels
    is_lda : bool, default False
        Whether to create LDA model (True) or ShallowFBCSPNet (False)
    random_state : int, optional
        Random state for reproducibility (not used for LDA)
    n_subjects : int, optional
        Number of subjects (only used when enable_subject_layer=True)
    enable_subject_layer : bool, optional
        Whether to enable subject-specific input layer (only for ShallowFBCSPNet)
        If None, uses global config use_subject_layer
        
    Returns
    -------
    model : sklearn.LinearDiscriminantAnalysis or torch.nn.Module
    """
    if is_lda:
        return LDA()
    else:
        # Determine if subject layer should be enabled
        if enable_subject_layer is None:
            enable_subject_layer = use_subject_layer
        
        base_model = ShallowFBCSPNet(
            in_chans=n_channels,
            n_classes=N_CLASSES,
            input_window_samples=INPUT_WINDOW_SAMPLES,
            final_conv_length='auto'  # Let model auto-calculate based on input
        )
        
        # Add subject layer if enabled and we have subject information
        if enable_subject_layer and n_subjects is not None and n_subjects > 1:
            subject_layer = SubjectInputLayer(n_subjects, n_channels)
            return ShallowFBCSPNetWithSubjectLayer(subject_layer, base_model)
        else:
            return base_model


def normalize_data(x):
    """
    Normalize data by z-score normalization across time dimension.
    """
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True) + NORMALIZATION_EPSILON
    return (x - mean) / std


def early_stopping(val_acc, model, state, patience = EARLY_STOPPING_PATIENCE):
    if 'best_val_acc' not in state:
        state['best_val_acc'] = 0
        state['counter'] = 0
        state['best_model'] = None
        state['early_stop'] = False

    if val_acc > state['best_val_acc']:
        state['best_val_acc'] = val_acc
        state['counter'] = 0
        state['best_model'] = model.state_dict().copy()
    else:
        state['counter'] += 1
        if state['counter'] >= patience:
            state['early_stop'] = True
    return state['early_stop']


def evaluate(model, loader, device, is_lda=False, subject_mapping=None):
    if is_lda:
        X = []
        y = []
        for batch_data in loader:
            if len(batch_data) == 3:  # (X, y, subject_indices)
                batch_X, batch_y, _ = batch_data
            else:  # (X, y)
                batch_X, batch_y = batch_data
            X.append(batch_X.reshape(batch_X.shape[0], -1).numpy())
            y.append(batch_y.numpy())
        X = np.concatenate(X)
        y = np.concatenate(y)
        predictions = model.predict(X)
        return np.mean(predictions == y)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 3:  # (X, y, subject_indices)
                x, y, subject_indices = batch_data
                subject_indices = subject_indices.to(device)
            else:  # (X, y) - backward compatibility
                x, y = batch_data
                subject_indices = None
            
            x = normalize_data(x).to(device)
            y = y.to(device)
            
            # Fix: Ensure y is 1D (class indices, not one-hot)
            if y.ndim > 1:
                y = torch.argmax(y, dim=1)
            
            # Forward pass with subject indices if model supports it
            if hasattr(model, 'subject_layer') and subject_indices is not None:
                scores = model(x, subject_indices)
            else:
                scores = model(x)
            
            if scores.ndim > 2:
                scores = scores.view(scores.size(0), -1)  
            
            _, predicted = scores.max(1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    
    return correct / total


def train_model(model, train_loader, val_loader, test_loader, device, is_lda=False, max_epochs=MAX_EPOCHS):
    if is_lda:
        # Prepare data for LDA
        X_train = []
        y_train = []
        for batch_data in train_loader:
            if len(batch_data) == 3:  # (X, y, subject_indices)
                batch_X, batch_y, _ = batch_data
            else:  # (X, y)
                batch_X, batch_y = batch_data
            X_train.append(batch_X.reshape(batch_X.shape[0], -1).numpy())
            y_train.append(batch_y.numpy())
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        
        # Train LDA model
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        return evaluate(model, test_loader, device, is_lda=True)
    
    # Neural Network training
    optimizer = torch.optim.Adamax(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)
    # Maintain state for early stopping using the helper function defined above
    es_state = {}

    for epoch in range(max_epochs):
        model.train()
        for batch_data in train_loader:
            if len(batch_data) == 3:  # (X, y, subject_indices)
                x, y, subject_indices = batch_data
                subject_indices = subject_indices.to(device)
            else:  # (X, y) - backward compatibility
                x, y = batch_data
                subject_indices = None
            
            x = normalize_data(x).to(device)
            y = y.to(device)
            
            if y.ndim > 1:
                y = torch.argmax(y, dim=1)
            elif y.ndim == 1:
                y = y.long()
            
            optimizer.zero_grad()
            
            # Forward pass with subject indices if model supports it
            if hasattr(model, 'subject_layer') and subject_indices is not None:
                scores = model(x, subject_indices)
            else:
                scores = model(x)

            if scores.ndim > 2:
                scores = scores.view(scores.size(0), -1)
            
            loss = F.cross_entropy(scores, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Validation phase
        val_acc = evaluate(model, val_loader, device)
        
        # Early stopping check
        if early_stopping(val_acc, model, es_state, patience = EARLY_STOPPING_PATIENCE):
            break
    
    # Load best model and evaluate on test set
    if 'best_model' in es_state and es_state['best_model'] is not None:
        model.load_state_dict(es_state['best_model'])
    return evaluate(model, test_loader, device) 