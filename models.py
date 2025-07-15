"""
Model definitions and related functions for EEG experiments
"""

import torch
import torch.nn.functional as F
import numpy as np
from braindecode.models import ShallowFBCSPNet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from config import INPUT_WINDOW_SAMPLES


def create_model(n_channels, is_lda=False, random_state=None):
    """Create a new model based on configuration.
    
    Parameters
    ----------
    n_channels : int
        Number of input channels
    is_lda : bool, default False
        Whether to create LDA model (True) or ShallowFBCSPNet (False)
    random_state : int, optional
        Random state for reproducibility (not used for LDA)
        
    Returns
    -------
    model : sklearn.LinearDiscriminantAnalysis or braindecode.models.ShallowFBCSPNet
    """
    if is_lda:
        return LDA()
    else:
        return ShallowFBCSPNet(
            in_chans=n_channels,
            n_classes=2,
            input_window_samples=INPUT_WINDOW_SAMPLES,
            final_conv_length='auto'  # Let model auto-calculate based on input
        )


def normalize_data(x):
    """
    Normalize data by z-score normalization across time dimension.
    """
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True) + 1e-7
    return (x - mean) / std


def early_stopping(val_acc, model, state, patience=5):
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


def evaluate(model, loader, device, is_lda=False):
    if is_lda:
        X = []
        y = []
        for batch_X, batch_y in loader:
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
        for x, y in loader:
            x = normalize_data(x).to(device)
            y = y.to(device)
            
            # Fix: Ensure y is 1D (class indices, not one-hot)
            if y.ndim > 1:
                y = torch.argmax(y, dim=1)
            
            scores = model(x)
            
            if scores.ndim > 2:
                scores = scores.view(scores.size(0), -1)  
            
            _, predicted = scores.max(1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    
    return correct / total


def train_model(model, train_loader, val_loader, test_loader, device, is_lda=False, max_epochs=100):
    if is_lda:
        # Prepare data for LDA
        X_train = []
        y_train = []
        for batch_X, batch_y in train_loader:
            X_train.append(batch_X.reshape(batch_X.shape[0], -1).numpy())
            y_train.append(batch_y.numpy())
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        
        # Train LDA model
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        return evaluate(model, test_loader, device, is_lda=True)
    
    # Neural Network training
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.005, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    # Maintain state for early stopping using the helper function defined above
    es_state = {}

    for epoch in range(max_epochs):
        model.train()
        for x, y in train_loader:
            x = normalize_data(x).to(device)
            y = y.to(device)
            
            if y.ndim > 1:
                y = torch.argmax(y, dim=1)
            elif y.ndim == 1:
                y = y.long()
            
            optimizer.zero_grad()
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
        if early_stopping(val_acc, model, es_state, patience=10):
            break
    
    # Load best model and evaluate on test set
    if 'best_model' in es_state and es_state['best_model'] is not None:
        model.load_state_dict(es_state['best_model'])
    return evaluate(model, test_loader, device) 