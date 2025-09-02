"""
Model definitions and related functions for EEG experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from braindecode.models import ShallowFBCSPNet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from config import (
    INPUT_WINDOW_SAMPLES, use_subject_layer, EARLY_STOPPING_PATIENCE,
    LEARNING_RATE, WEIGHT_DECAY, GAMMA, MAX_EPOCHS, N_CLASSES,
    USE_DATA_AUGMENTATION, NOISE_STD, TIME_SHIFT_RANGE, LABEL_SMOOTHING, DROPOUT_RATE
)
from constants import NORMALIZATION_EPSILON


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def augment_data(x, training=True):
    """Apply data augmentation to EEG data."""
    if not training or not USE_DATA_AUGMENTATION:
        return x
    
    batch_size, n_channels, n_timepoints = x.shape
    augmented_x = x.clone()
    
    # Add Gaussian noise
    if NOISE_STD > 0:
        noise = torch.randn_like(augmented_x) * NOISE_STD
        augmented_x = augmented_x + noise
    
    # Time shifting
    if TIME_SHIFT_RANGE > 0:
        for i in range(batch_size):
            shift = np.random.randint(-TIME_SHIFT_RANGE, TIME_SHIFT_RANGE + 1)
            if shift != 0:
                if shift > 0:
                    augmented_x[i, :, shift:] = x[i, :, :-shift]
                    augmented_x[i, :, :shift] = x[i, :, -shift:]
                else:
                    augmented_x[i, :, :shift] = x[i, :, -shift:]
                    augmented_x[i, :, shift:] = x[i, :, :-shift]
    
    return augmented_x


def label_smoothing_loss(pred, target, smoothing=LABEL_SMOOTHING):
    """Compute label smoothing loss."""
    if smoothing == 0.0:
        return F.cross_entropy(pred, target)
    
    n_classes = pred.size(-1)
    one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
    smooth_one_hot = one_hot * (1 - smoothing) + smoothing / n_classes
    return -(smooth_one_hot * F.log_softmax(pred, dim=1)).sum(dim=1).mean()


class SubjectInputLayer(nn.Module):
    """Layer that applies subject-specific linear transformations to input data."""
    def __init__(self, n_subjects, n_channels):
        super().__init__()
        # Initialize with identity matrices (no transformation initially)
        self.weights = nn.Parameter(torch.eye(n_channels).unsqueeze(0).repeat(n_subjects, 1, 1))
        self.n_subjects = n_subjects
        self.n_channels = n_channels
    
    def forward(self, x, subject_indices):
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
    is_lda : bool, default False
    n_subjects : int, optional
    enable_subject_layer : bool, optional
        
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
            n_chans=n_channels,
            n_outputs=N_CLASSES,
            n_times=INPUT_WINDOW_SAMPLES,
            final_conv_length='auto'  
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


def evaluate(model, loader, device, is_lda=False, subject_mapping=None, return_details=False):
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
        correct_count = np.sum(predictions == y)
        total_count = len(y)
        accuracy = correct_count / total_count
        
        if return_details:
            try:
                # Get probability estimates for AUC calculation
                y_proba = model.predict_proba(X)[:, 1]  # Probability of positive class
            except:
                y_proba = predictions  # Fallback to binary predictions if probabilities not available
            
            # Calculate metrics
            precision = precision_score(y, predictions, average='binary', zero_division=0)
            recall = recall_score(y, predictions, average='binary', zero_division=0)
            f1 = f1_score(y, predictions, average='binary', zero_division=0)
            try:
                auc = roc_auc_score(y, y_proba)
            except:
                auc = 0.5  # Default AUC if calculation fails
            
            return {
                'accuracy': accuracy,
                'correct_count': correct_count,
                'incorrect_count': total_count - correct_count,
                'total_count': total_count,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc
            }
        return accuracy
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 3:  
                x, y, subject_indices = batch_data
                subject_indices = subject_indices.to(device)
            else: 
                x, y = batch_data
                subject_indices = None
            
            x = normalize_data(x).to(device)
            y = y.to(device)
            
           
            if y.ndim > 1:
                y = torch.argmax(y, dim=1)

            if hasattr(model, 'subject_layer') and subject_indices is not None:
                scores = model(x, subject_indices)
            else:
                scores = model(x)
            
            if scores.ndim > 2:
                scores = scores.view(scores.size(0), -1)  
            
            _, predicted = scores.max(1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            
            # Store predictions and targets for detailed metrics
            if return_details:
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                # Store probabilities for AUC calculation
                probabilities = F.softmax(scores, dim=1)[:, 1]  # Probability of positive class
                all_probabilities.extend(probabilities.cpu().numpy())
    
    accuracy = correct / total
    if return_details:
        # Calculate precision, recall, F1 score and AUC
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        precision = precision_score(all_targets, all_predictions, average='binary', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='binary', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='binary', zero_division=0)
        
        # Calculate AUC
        try:
            auc = roc_auc_score(all_targets, all_probabilities)
        except:
            auc = 0.5  # Default AUC if calculation fails
        
        return {
            'accuracy': accuracy,
            'correct_count': correct,
            'incorrect_count': total - correct,
            'total_count': total,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
    return accuracy


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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    # Maintain state for early stopping using the helper function defined above
    es_state = {}

    # Compute effective class weights using effective number of samples
    class_weights = None
    try:
        if hasattr(train_loader.dataset, 'tensors'):
            y_all = train_loader.dataset.tensors[1]
        elif hasattr(train_loader.dataset, 'labels'):
            y_all = train_loader.dataset.labels
        else:
            y_all = None
        if y_all is not None:
            y_np = y_all.detach().cpu().numpy()
            num_classes = int(y_np.max()) + 1
            counts = np.bincount(y_np, minlength=num_classes)
            
            # Use effective number of samples for better class weighting
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, counts)
            weights_np = (1.0 - beta) / np.array(effective_num)
            weights_np = weights_np / np.sum(weights_np) * num_classes
            
            class_weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
            print(f"Training class distribution: {counts.tolist()} | effective class weights: {weights_np.tolist()}")
    except Exception as e:
        print(f"Warning: failed to compute class weights: {e}")
        class_weights = None

    # Initialize focal loss
    focal_loss = FocalLoss(alpha=1, gamma=2, weight=class_weights)

    for epoch in range(max_epochs):
        model.train()
        for batch_data in train_loader:
            if len(batch_data) == 3:  # (X, y, subject_indices)
                x, y, subject_indices = batch_data
                subject_indices = subject_indices.to(device)
            else:  # (X, y) - backward compatibility
                x, y = batch_data
                subject_indices = None
            
            # Apply data augmentation
            x = augment_data(x, training=True)
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
            
            # Use focal loss with label smoothing
            if LABEL_SMOOTHING > 0:
                loss = label_smoothing_loss(scores, y, LABEL_SMOOTHING)
            else:
                loss = focal_loss(scores, y)
            
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
