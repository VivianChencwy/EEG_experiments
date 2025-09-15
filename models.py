"""
Model definitions and related functions for EEG experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
try:
    from braindecode.models import ShallowFBCSPNet
    BRAINDECODE_AVAILABLE = True
except (ImportError, AttributeError, Exception):
    BRAINDECODE_AVAILABLE = False
    # Define a dummy ShallowFBCSPNet to avoid reference errors
    ShallowFBCSPNet = None
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import math

from config import (
    INPUT_WINDOW_SAMPLES, use_subject_layer, EARLY_STOPPING_PATIENCE,
    LEARNING_RATE, WEIGHT_DECAY, GAMMA, MAX_EPOCHS, N_CLASSES,
    USE_DATA_AUGMENTATION, NOISE_STD, TIME_SHIFT_RANGE, LABEL_SMOOTHING, DROPOUT_RATE,
    ELECTRODE_FUSION_METHOD, DOMAIN_ADAPTATION_METHOD
)
from constants import NORMALIZATION_EPSILON


class CustomShallowFBCSPNet(nn.Module):
    """Custom implementation of ShallowFBCSPNet."""
    def __init__(self, n_chans, n_outputs, n_times, final_conv_length='auto'):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        
        # Temporal convolution
        self.temporal_conv = nn.Conv2d(1, 40, (1, 25), padding=(0, 12))
        
        # Spatial convolution
        self.spatial_conv = nn.Conv2d(40, 40, (n_chans, 1), bias=False)
        self.bn = nn.BatchNorm2d(40)
        
        # Pooling
        self.pool = nn.AvgPool2d((1, 75), (1, 15))
        
        # Calculate output size
        self._calculate_final_conv_length()
        
        # Final classification layer
        self.classifier = nn.Linear(self.final_length, n_outputs)
        
    def _calculate_final_conv_length(self):
        # Calculate the final convolution length
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_chans, self.n_times)
            x = self.temporal_conv(x)  
            x = self.spatial_conv(x)   
            x = self.bn(x)             
            x = F.elu(x)               
            x = self.pool(x)           
            self.final_length = x.numel() // x.size(0)
    
    def forward(self, x):
        # x shape: (batch, n_chans, n_times)
        x = x.unsqueeze(1)  # (batch, 1, n_chans, n_times)
        
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = F.elu(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class EEGNet(nn.Module):
    """EEGNet implementation for EEG classification."""
    def __init__(self, n_chans, n_outputs, n_times, 
                 F1=8, F2=16, D=2, dropout=0.5):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.F1 = F1
        self.F2 = F2
        self.D = D
        
        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(F1, F1*D, (n_chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D)
        
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        
        # Block 2
        # Separable convolution
        self.separable_conv = nn.Conv2d(F1*D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)
        
        # Calculate final dimensions
        self._calculate_final_dims(n_times)
        
        # Classification
        self.classifier = nn.Linear(self.final_length, n_outputs)
        
    def _calculate_final_dims(self, n_times):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_chans, n_times)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.depthwise_conv(x)
            x = self.bn2(x)
            x = F.elu(x)
            x = self.pool1(x)
            x = self.dropout1(x)
            
            x = self.separable_conv(x)
            x = self.bn3(x)
            x = F.elu(x)
            x = self.pool2(x)
            x = self.dropout2(x)
            
            self.final_length = x.numel() // x.size(0)
    
    def forward(self, x):
        # x shape: (batch, n_chans, n_times)
        x = x.unsqueeze(1)  # (batch, 1, n_chans, n_times)
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.separable_conv(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Classification
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class DeepConvNet(nn.Module):
    """Deep Convolutional Network for EEG."""
    def __init__(self, n_chans, n_outputs, n_times, dropout=0.5):
        super().__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 25, (1, 10))
        self.conv2 = nn.Conv2d(25, 25, (n_chans, 1))
        self.bn1 = nn.BatchNorm2d(25)
        self.pool1 = nn.MaxPool2d((1, 3))
        
        # Block 2
        self.conv3 = nn.Conv2d(25, 50, (1, 10))
        self.bn2 = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d((1, 3))
        
        # Block 3
        self.conv4 = nn.Conv2d(50, 100, (1, 10))
        self.bn3 = nn.BatchNorm2d(100)
        self.pool3 = nn.MaxPool2d((1, 3))
        
        # Block 4
        self.conv5 = nn.Conv2d(100, 200, (1, 10))
        self.bn4 = nn.BatchNorm2d(200)
        self.pool4 = nn.MaxPool2d((1, 3))
        
        self.dropout = nn.Dropout(dropout)
        
        # Calculate final dimensions
        self._calculate_final_dims(n_times)
        
        # Classification
        self.classifier = nn.Linear(self.final_length, n_outputs)
        
    def _calculate_final_dims(self, n_times):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_chans, n_times)
            x = self._forward_features(x)
            self.final_length = x.numel() // x.size(0)
    
    def _forward_features(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        # Block 2
        x = self.conv3(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout(x)
        
        # Block 3
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.dropout(x)
        
        # Block 4
        x = self.conv5(x)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.pool4(x)
        x = self.dropout(x)
        
        return x
    
    def forward(self, x):
        # x shape: (batch, n_chans, n_times)
        x = x.unsqueeze(1)  # (batch, 1, n_chans, n_times)
        
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


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


class EEGConformer(nn.Module):
    """EEGConformer: Combining CNN and Transformer for EEG classification."""
    def __init__(self, n_chans, n_outputs, n_times, 
                 conv_spatial_dim=40, conv_temporal_dim=25,
                 embedding_dim=40, num_heads=10, num_layers=3,
                 dropout=0.5, activation='gelu'):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.embedding_dim = embedding_dim
        
        # Temporal convolution
        self.temporal_conv = nn.Conv2d(1, conv_temporal_dim, (1, 25), padding=(0, 12))
        self.temporal_bn = nn.BatchNorm2d(conv_temporal_dim)
        
        # Spatial convolution  
        self.spatial_conv = nn.Conv2d(conv_temporal_dim, conv_spatial_dim, (n_chans, 1))
        self.spatial_bn = nn.BatchNorm2d(conv_spatial_dim)
        
        # Pooling and dropout
        self.avg_pool = nn.AvgPool2d((1, 4), (1, 4))
        self.dropout = nn.Dropout(dropout)
        
        # Calculate sequence length after convolutions
        seq_length = self._get_sequence_length()
        
        # Projection to embedding dimension
        self.projection = nn.Linear(conv_spatial_dim, embedding_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, max_len=seq_length)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(embedding_dim, n_outputs)
        )
    
    def _get_sequence_length(self):
        # Calculate sequence length after convolutions
        # After temporal conv: n_times (same due to padding)
        # After avg pool: n_times // 4
        return self.n_times // 4
    
    def forward(self, x):
        # x shape: (batch, n_chans, n_times)
        x = x.unsqueeze(1)  # (batch, 1, n_chans, n_times)
        
        # Temporal convolution
        x = self.temporal_conv(x)  # (batch, conv_temporal_dim, n_chans, n_times)
        x = self.temporal_bn(x)
        x = F.elu(x)
        
        # Spatial convolution
        x = self.spatial_conv(x)  # (batch, conv_spatial_dim, 1, n_times)
        x = self.spatial_bn(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Pooling
        x = self.avg_pool(x)  # (batch, conv_spatial_dim, 1, n_times//4)
        
        # Reshape for transformer
        x = x.squeeze(2).transpose(1, 2)  # (batch, seq_len, conv_spatial_dim)
        
        # Project to embedding dimension
        x = self.projection(x)  # (batch, seq_len, embedding_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer
        x = self.transformer(x)  # (batch, seq_len, embedding_dim)
        
        # Classification
        x = x.transpose(1, 2)  # (batch, embedding_dim, seq_len)
        x = self.classifier(x)  # (batch, n_outputs)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EEGChannelNet(nn.Module):
    """Advanced EEG model with channel-wise attention."""
    def __init__(self, n_chans, n_outputs, n_times, dropout=0.5):
        super().__init__()
        
        # Channel-wise convolution
        self.channel_conv = nn.ModuleList([
            nn.Conv1d(1, 8, kernel_size=64, padding=32) for _ in range(n_chans)
        ])
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.Linear(n_chans * 8, n_chans),
            nn.Sigmoid()
        )
        
        # Temporal convolution layers
        self.conv1 = nn.Conv2d(1, 16, (n_chans, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, (1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 16))
        self.dropout = nn.Dropout(dropout)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_outputs)
        )
    
    def forward(self, x):
        # x shape: (batch, n_chans, n_times)
        batch_size = x.size(0)
        
        # Channel-wise processing
        channel_features = []
        for i, conv in enumerate(self.channel_conv):
            chan_out = conv(x[:, i:i+1, :])  # (batch, 8, n_times)
            channel_features.append(chan_out.mean(dim=2))  # (batch, 8)
        
        # Channel attention
        channel_features = torch.stack(channel_features, dim=1)  # (batch, n_chans, 8)
        channel_features = channel_features.view(batch_size, -1)  # (batch, n_chans*8)
        attention_weights = self.channel_attention(channel_features)  # (batch, n_chans)
        
        # Apply attention to input
        x = x * attention_weights.unsqueeze(2)  # (batch, n_chans, n_times)
        
        # Main processing
        x = x.unsqueeze(1)  # (batch, 1, n_chans, n_times)
        
        x = self.conv1(x)  # (batch, 16, 1, n_times)
        x = self.bn1(x)
        x = F.elu(x)
        
        x = self.conv2(x)  # (batch, 32, 1, n_times)
        x = self.bn2(x)
        x = F.elu(x)
        
        x = self.conv3(x)  # (batch, 64, 1, n_times)
        x = self.bn3(x)
        x = F.elu(x)
        
        x = self.pool(x)  # (batch, 64, 1, 16)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)  # (batch, 64*16)
        x = self.classifier(x)
        
        return x


def create_model(n_channels, is_lda=False, random_state=None, n_subjects=None, enable_subject_layer=None, model_name='ShallowFBCSPNet', input_channels=None):
    """Create a new model based on configuration.
    
    Parameters
    ----------
    n_channels : int
    is_lda : bool, default False
    n_subjects : int, optional
    enable_subject_layer : bool, optional
    model_name : str, default 'ShallowFBCSPNet'
        Options: 'ShallowFBCSPNet', 'EEGNetv4', 'Deep4Net', 'EEGConformer', 'EEGChannelNet'
    input_channels : int, optional
        Actual number of input channels (may differ from n_channels if features were added)
        
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

        # Use input_channels if provided, otherwise use n_channels
        actual_channels = input_channels if input_channels is not None else n_channels
        
        # Create base model based on model_name
        if model_name == 'ShallowFBCSPNet':
            if BRAINDECODE_AVAILABLE:
                base_model = ShallowFBCSPNet(
                    n_chans=actual_channels,
                    n_outputs=N_CLASSES,
                    n_times=INPUT_WINDOW_SAMPLES,
                    final_conv_length='auto'
                )
            else:
                base_model = CustomShallowFBCSPNet(
                    n_chans=actual_channels,
                    n_outputs=N_CLASSES,
                    n_times=INPUT_WINDOW_SAMPLES
                )
        elif model_name == 'EEGNet' or model_name == 'EEGNetv4':
            base_model = EEGNet(
                n_chans=actual_channels,
                n_outputs=N_CLASSES,
                n_times=INPUT_WINDOW_SAMPLES,
                dropout=DROPOUT_RATE
            )
        elif model_name == 'DeepConvNet' or model_name == 'Deep4Net':
            base_model = DeepConvNet(
                n_chans=actual_channels,
                n_outputs=N_CLASSES,
                n_times=INPUT_WINDOW_SAMPLES,
                dropout=DROPOUT_RATE
            )
        elif model_name == 'EEGConformer':
            # Get EEGConformer parameters from config if available
            try:
                from config import (
                    CONFORMER_CONV_SPATIAL_DIM, CONFORMER_CONV_TEMPORAL_DIM,
                    CONFORMER_EMBEDDING_DIM, CONFORMER_NUM_HEADS,
                    CONFORMER_NUM_LAYERS, CONFORMER_ACTIVATION
                )
            except ImportError:
                # Default parameters
                CONFORMER_CONV_SPATIAL_DIM = 40
                CONFORMER_CONV_TEMPORAL_DIM = 25
                CONFORMER_EMBEDDING_DIM = 40
                CONFORMER_NUM_HEADS = 10
                CONFORMER_NUM_LAYERS = 3
                CONFORMER_ACTIVATION = 'gelu'

            base_model = EEGConformer(
                n_chans=actual_channels,
                n_outputs=N_CLASSES,
                n_times=INPUT_WINDOW_SAMPLES,
                conv_spatial_dim=CONFORMER_CONV_SPATIAL_DIM,
                conv_temporal_dim=CONFORMER_CONV_TEMPORAL_DIM,
                embedding_dim=CONFORMER_EMBEDDING_DIM,
                num_heads=CONFORMER_NUM_HEADS,
                num_layers=CONFORMER_NUM_LAYERS,
                dropout=DROPOUT_RATE,
                activation=CONFORMER_ACTIVATION
            )
        elif model_name == 'EEGChannelNet':
            base_model = EEGChannelNet(
                n_chans=actual_channels,
                n_outputs=N_CLASSES,
                n_times=INPUT_WINDOW_SAMPLES,
                dropout=DROPOUT_RATE
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Add subject layer if enabled and we have subject information
        # Note: Subject layer only works with ShallowFBCSPNet for now
        if (enable_subject_layer and n_subjects is not None and n_subjects > 1 
            and model_name == 'ShallowFBCSPNet'):
            subject_layer = SubjectInputLayer(n_subjects, n_channels)
            return ShallowFBCSPNetWithSubjectLayer(subject_layer, base_model)
        else:
            return base_model


def normalize_data(x):
    """
    Normalize data with robust handling of constant channels and enhanced features.
    Normalizes across time dimension (dim=2) for each channel independently.
    """
    # Debug: Check input data
    if torch.all(x == 0):
        print("WARNING: All input data to normalize_data is zero!")
        return x

    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True)

    # More robust handling of zero standard deviation
    zero_std_mask = (std <= NORMALIZATION_EPSILON)
    num_zero_std = torch.sum(zero_std_mask).item()

    if num_zero_std > 0:
        # Only warn once per batch if many channels have zero std
        # if num_zero_std > x.shape[1] * 0.1:  # More than 10% of channels
        #     print(f"INFO: {num_zero_std}/{x.shape[1]} channels have near-zero variance (likely constant features)")

        # For constant channels, keep them as-is (subtract mean, but don't divide by std)
        # This is better than setting std=1 which can create artificial scaling
        std = torch.where(zero_std_mask, torch.ones_like(std), std)

    # Apply normalization
    std = std + NORMALIZATION_EPSILON
    normalized = (x - mean) / std

    # For originally constant channels, set them to zero (mean-centered)
    normalized = torch.where(zero_std_mask.expand_as(normalized),
                           torch.zeros_like(normalized), normalized)

    # Final check for numerical issues
    if torch.any(torch.isnan(normalized)) or torch.any(torch.isinf(normalized)):
        print("WARNING: NaN or Inf values after normalization, cleaning...")
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)

    return normalized


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
        
        if return_details:
            try:
                # Get probability estimates for AUC calculation
                y_proba = model.predict_proba(X)[:, 1]  # Probability of positive class
            except:
                y_proba = predictions  # Fallback to binary predictions if probabilities not available
            
            # Calculate confusion matrix first
            cm = confusion_matrix(y, predictions)
            
            # Handle different confusion matrix shapes
            if cm.shape == (1, 1):
                # Only one class present
                tp = cm[0, 0] if predictions[0] == y[0] else 0
                tn = fp = fn = 0
                accuracy = 1.0 if tp > 0 else 0.0
                precision = recall = f1 = 1.0 if tp > 0 else 0.0
            elif cm.shape == (2, 2):
                # Standard 2x2 confusion matrix
                tn, fp, fn, tp = cm.ravel()
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            else:
                # Fallback: calculate metrics directly
                correct = np.sum(predictions == y)
                accuracy = correct / len(y)
                tp = tn = fp = fn = 0
                precision = recall = f1 = 0.0
            try:
                # Check if we have both classes in the true labels
                unique_labels = np.unique(y)
                if len(unique_labels) < 2:
                    print(f"Warning: Only one class present in test set: {unique_labels}. Setting AUC to 0.5.")
                    auc = 0.5
                else:
                    # Check for problematic probability values
                    if np.any(np.isnan(y_proba)) or np.any(np.isinf(y_proba)):
                        print(f"Warning: Found NaN or infinite values in probabilities. Setting AUC to 0.5.")
                        auc = 0.5
                    else:
                        auc = roc_auc_score(y, y_proba)
                        if np.isnan(auc):
                            print(f"Warning: AUC calculation returned NaN. Setting to 0.5.")
                            auc = 0.5
            except Exception as e:
                print(f"Warning: AUC calculation failed: {e}. Setting to 0.5.")
                auc = 0.5
            
            return {
                'accuracy': accuracy,
                'correct_count': tp + tn,
                'incorrect_count': fp + fn,
                'total_count': tp + tn + fp + fn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            }
        # For LDA without details, calculate accuracy from confusion matrix
        cm = confusion_matrix(y, predictions)
        
        # Handle different confusion matrix shapes
        if cm.shape == (1, 1):
            # Only one class present
            return 1.0 if predictions[0] == y[0] else 0.0
        elif cm.shape == (2, 2):
            # Standard 2x2 confusion matrix
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            return accuracy
        else:
            # Fallback: calculate accuracy directly
            correct = np.sum(predictions == y)
            return correct / len(y)
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    correct = 0
    total = 0
    
    # Debug: Check loader
    loader_size = len(loader.dataset)
    if loader_size == 0:
        print(f"Warning: Loader is empty in evaluate function!")
        return 0.0
    
    with torch.no_grad():
        batch_count = 0
        for batch_data in loader:
            batch_count += 1
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
            
            # Collect predictions and targets for detailed evaluation
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
            # Get probabilities for AUC calculation
            probabilities = torch.softmax(scores, dim=1)
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of positive class
    
    if return_details:
        # Calculate precision, recall, F1 score and AUC
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate confusion matrix first
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Handle different confusion matrix shapes
        if cm.shape == (1, 1):
            # Only one class present
            tp = cm[0, 0] if all_predictions[0] == all_targets[0] else 0
            tn = fp = fn = 0
            accuracy = 1.0 if tp > 0 else 0.0
            precision = recall = f1 = 1.0 if tp > 0 else 0.0
        elif cm.shape == (2, 2):
            # Standard 2x2 confusion matrix
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            # Fallback: calculate metrics directly
            correct = np.sum(all_predictions == all_targets)
            accuracy = correct / len(all_targets)
            tp = tn = fp = fn = 0
            precision = recall = f1 = 0.0
        
        # Calculate AUC
        try:
            # Check if we have both classes in the true labels
            unique_labels = np.unique(all_targets)
            if len(unique_labels) < 2:
                print(f"Warning: Only one class present in overall test set: {unique_labels}. Setting AUC to 0.5.")
                auc = 0.5
            else:
                # Check for problematic probability values
                if np.any(np.isnan(all_probabilities)) or np.any(np.isinf(all_probabilities)):
                    print(f"Warning: Found NaN or infinite values in overall probabilities. Setting AUC to 0.5.")
                    auc = 0.5
                else:
                    auc = roc_auc_score(all_targets, all_probabilities)
                    if np.isnan(auc):
                        print(f"Warning: Overall AUC calculation returned NaN. Setting to 0.5.")
                        auc = 0.5
        except Exception as e:
            print(f"Warning: Overall AUC calculation failed: {e}. Setting to 0.5.")
            auc = 0.5
        
        return {
            'accuracy': accuracy,
            'correct_count': tp + tn,
            'incorrect_count': fp + fn,
            'total_count': tp + tn + fp + fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    # For neural network without details, calculate accuracy from confusion matrix
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Debug: Print evaluation info
    print(f"DEBUG: Evaluate function - {len(all_predictions)} predictions, {len(all_targets)} targets")
    if len(all_predictions) > 0:
        unique_preds = np.unique(all_predictions)
        unique_targets = np.unique(all_targets)
        print(f"DEBUG: Unique predictions: {unique_preds}, Unique targets: {unique_targets}")
    
    # Check if we have predictions and targets
    if len(all_predictions) == 0 or len(all_targets) == 0:
        print(f"Warning: No predictions or targets in evaluate function!")
        return 0.0
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Handle case where confusion matrix is not 2x2 (single class)
    if cm.shape == (1, 1):
        # Only one class present
        return 1.0 if all_predictions[0] == all_targets[0] else 0.0
    elif cm.shape == (2, 2):
        # Standard 2x2 confusion matrix
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return accuracy
    else:
        # Fallback: calculate accuracy directly
        correct = np.sum(all_predictions == all_targets)
        return correct / len(all_targets)


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

    # Since dataset is now balanced at source, no need for class weights
    try:
        if hasattr(train_loader.dataset, 'tensors'):
            y_all = train_loader.dataset.tensors[1]
        elif hasattr(train_loader.dataset, 'labels'):
            y_all = train_loader.dataset.labels
        else:
            y_all = None
        if y_all is not None:
            y_np = y_all.detach().cpu().numpy()
            counts = np.bincount(y_np)
            print(f"Training class distribution: {counts.tolist()}")
    except Exception as e:
        print(f"Warning: failed to get class distribution: {e}")

    # Initialize focal loss without class weights since dataset is balanced
    focal_loss = FocalLoss(alpha=1, gamma=2, weight=None)
    
    # Training progress tracking
    print(f"\n{'='*60}")
    print(f"Starting Training - Max Epochs: {max_epochs}")
    print(f"Model: {type(model).__name__}")
    print(f"Learning Rate: {LEARNING_RATE}, Weight Decay: {WEIGHT_DECAY}")
    print(f"Dropout: {DROPOUT_RATE}, Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print(f"{'='*60}")

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        batch_count = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
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
            
            # Track training statistics
            epoch_loss += loss.item()
            _, predicted = scores.max(1)
            epoch_correct += (predicted == y).sum().item()
            epoch_total += y.size(0)
            batch_count += 1
            
        
        # Calculate epoch statistics
        avg_loss = epoch_loss / batch_count
        train_acc = 100. * epoch_correct / epoch_total
        current_lr = optimizer.param_groups[0]['lr']
        
        scheduler.step()
        
        # Validation phase
        # Debug: Check validation loader
        val_samples = len(val_loader.dataset)
        if val_samples == 0:
            print(f"Warning: Validation loader is empty!")
            val_acc = 0.0
        else:
            val_acc = evaluate(model, val_loader, device)
            if val_acc == 0.0 and val_samples > 0:
                print(f"Warning: Validation accuracy is 0.0 with {val_samples} samples")
        val_acc_percent = 100. * val_acc
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1:3d}/{max_epochs} Summary:")
        print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc_percent:.2f}% | LR: {current_lr:.6f}")
        
        # Early stopping check with detailed info
        is_best = False
        if 'best_val_acc' not in es_state or val_acc > es_state['best_val_acc']:
            is_best = True
            
        if early_stopping(val_acc, model, es_state, patience = EARLY_STOPPING_PATIENCE):
            print(f"Early stopping triggered! No improvement for {EARLY_STOPPING_PATIENCE} epochs")
            print(f"Best validation accuracy: {100. * es_state['best_val_acc']:.2f}%")
            break
        else:
            if is_best:
                print(f"New best validation accuracy!")
            else:
                remaining_patience = EARLY_STOPPING_PATIENCE - es_state['counter']
                print(f"Patience remaining: {remaining_patience}/{EARLY_STOPPING_PATIENCE}")
        
        print(f"  {'-'*50}")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    if 'best_val_acc' in es_state:
        print(f"Best Validation Accuracy: {100. * es_state['best_val_acc']:.2f}%")
    print(f"{'='*60}")
    
    # Load best model and evaluate on test set
    if 'best_model' in es_state and es_state['best_model'] is not None:
        model.load_state_dict(es_state['best_model'])
    return evaluate(model, test_loader, device)


def create_fusion_model(model_name: str, datasets_info: Dict, fusion_method: str = 'none',
                       domain_adaptation: str = 'none', **kwargs):
    """
    Create a model with fusion and domain adaptation capabilities

    Args:
        model_name: Base model name
        datasets_info: Information about datasets
        fusion_method: Fusion method to use
        domain_adaptation: Domain adaptation method
        **kwargs: Additional parameters

    Returns:
        Model instance
    """
    # Import fusion methods here to avoid circular imports
    from fusion_methods import FusionModelFactory
    from domain_adaptation import DomainAdapterFactory

    if fusion_method == 'none' and domain_adaptation == 'none':
        # Use existing create_model function for baseline
        max_channels = max(len(info['channels']) for info in datasets_info.values()) if datasets_info else 16
        return create_model(model_name, max_channels, **kwargs)

    elif fusion_method in ['graph_gcn', 'spatial_attention']:
        # Create fusion model
        fusion_model = FusionModelFactory.create_fusion_model(
            fusion_method, datasets_info,
            base_model_info={
                'class': lambda **params: create_model(model_name, **params),
                'params': kwargs
            }
        )

        if domain_adaptation != 'none':
            # Wrap with domain adaptation
            feature_dim = 128  # This should be determined from the fusion model
            domain_adapter = DomainAdapterFactory.create_domain_adapter(
                domain_adaptation, fusion_model, feature_dim, datasets_info
            )
            return domain_adapter

        return fusion_model

    elif domain_adaptation != 'none':
        # Domain adaptation without fusion
        max_channels = max(len(info['channels']) for info in datasets_info.values()) if datasets_info else 16
        base_model = create_model(model_name, max_channels, **kwargs)

        feature_dim = 128  # This should be determined from the base model
        domain_adapter = DomainAdapterFactory.create_domain_adapter(
            domain_adaptation, base_model, feature_dim, datasets_info
        )
        return domain_adapter

    else:
        raise ValueError(f"Unsupported combination: fusion={fusion_method}, adaptation={domain_adaptation}")


def train_fusion_model(model, train_loaders: Dict, val_loaders: Dict, test_loaders: Dict,
                      device, fusion_method: str = 'none', domain_adaptation: str = 'none',
                      max_epochs: int = MAX_EPOCHS):
    """
    Train a fusion model with domain adaptation

    Args:
        model: Model to train
        train_loaders: Dictionary of training data loaders
        val_loaders: Dictionary of validation data loaders
        test_loaders: Dictionary of test data loaders
        device: Training device
        fusion_method: Fusion method being used
        domain_adaptation: Domain adaptation method
        max_epochs: Maximum training epochs

    Returns:
        Test accuracy
    """
    from domain_adaptation import DomainAdaptationLoss, DomainAdapterFactory
    from enhanced_preprocessor import MultiModalDataLoader

    if fusion_method == 'none' and domain_adaptation == 'none':
        # Use existing training function for baseline
        train_loader = list(train_loaders.values())[0]
        val_loader = list(val_loaders.values())[0]
        test_loader = list(test_loaders.values())[0]
        return train_model(model, train_loader, val_loader, test_loader, device, max_epochs=max_epochs)

    # For heterogeneous fusion, we'll iterate through each dataset separately
    # instead of trying to batch them together (which causes dimension mismatch)
    dataset_names = list(train_loaders.keys())
    train_iterators = {name: iter(loader) for name, loader in train_loaders.items()}

    # Move model to device
    model = model.to(device)

    # Setup optimizers based on adaptation method
    optimizers = DomainAdapterFactory.create_optimizers(model, domain_adaptation)

    # Setup loss function
    loss_fn = DomainAdaptationLoss(domain_adaptation)

    # Create learning rate schedulers
    schedulers = {}
    for name, optimizer in optimizers.items():
        schedulers[name] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Early stopping state
    es_state = {}

    print(f"\n{'='*60}")
    print(f"Starting Fusion Training - Max Epochs: {max_epochs}")
    print(f"Fusion Method: {fusion_method}")
    print(f"Domain Adaptation: {domain_adaptation}")
    print(f"Model: {type(model).__name__}")
    print(f"{'='*60}")

    for epoch in range(max_epochs):
        model.train()
        epoch_losses = {}
        epoch_correct = 0
        epoch_total = 0
        batch_count = 0

        # Training loop - process each dataset separately
        for batch_idx in range(50):  # Fixed number of batches per epoch
            try:
                # Cycle through datasets
                dataset_name = dataset_names[batch_idx % len(dataset_names)]

                # Get batch from current dataset
                try:
                    data, labels = next(train_iterators[dataset_name])
                except StopIteration:
                    # Reset iterator if exhausted
                    train_iterators[dataset_name] = iter(train_loaders[dataset_name])
                    data, labels = next(train_iterators[dataset_name])

                data, labels = data.to(device), labels.to(device)
                domains = [dataset_name] * len(data)  # Set domain for all samples in batch

                # Apply data augmentation and normalization
                data = augment_data(data, training=True)
                data = normalize_data(data)

                # Forward pass
                if domain_adaptation == 'adversarial':
                    # Adversarial training
                    alpha = 2.0 / (1.0 + np.exp(-10 * epoch / max_epochs)) - 1.0  # GRL schedule
                    task_pred, domain_pred, features = model(data, alpha=alpha, return_features=True)

                    # Create domain labels for current dataset
                    domain_labels = torch.full((len(data),), hash(dataset_name) % len(dataset_names),
                                             device=device, dtype=torch.long)
                    predictions = {'task': task_pred, 'domain': domain_pred}
                    targets = {'task': labels, 'domain': domain_labels}

                elif domain_adaptation == 'ms_mda':
                    # MS-MDA training - single domain per batch
                    domain = dataset_name
                    task_pred, shared_feat, adapted_feat = model(
                        data, domain=domain, return_features=True
                    )

                    # Compute adaptation loss (domain features will be accumulated across batches)
                    adaptation_loss = model.compute_adaptation_loss({domain: shared_feat})

                    predictions = {'task': task_pred, 'adaptation': adaptation_loss}
                    targets = {'task': labels}

                else:
                    # No domain adaptation
                    if hasattr(model, 'forward') and 'dataset_name' in model.forward.__code__.co_varnames:
                        task_pred = model(data, dataset_name=domains[0] if domains else 'unknown')
                    else:
                        task_pred = model(data)
                    predictions = {'task': task_pred}
                    targets = {'task': labels}

                # Compute loss
                total_loss, loss_components = loss_fn(predictions, targets, model)

                # Backward pass
                if domain_adaptation == 'adversarial' and 'discriminator' in optimizers:
                    # Adversarial training: alternate updates
                    if batch_idx % 2 == 0:
                        # Update feature extractor
                        optimizers['feature'].zero_grad()
                        total_loss.backward()
                        optimizers['feature'].step()
                    else:
                        # Update discriminator
                        optimizers['discriminator'].zero_grad()
                        loss_components['domain_loss'].backward()
                        optimizers['discriminator'].step()
                else:
                    # Standard training
                    optimizer = optimizers.get('main', list(optimizers.values())[0])
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                # Track statistics
                for key, value in loss_components.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0
                    epoch_losses[key] += value.item() if torch.is_tensor(value) else value

                # Calculate accuracy
                if torch.is_tensor(predictions['task']):
                    _, predicted = predictions['task'].max(1)
                    epoch_correct += (predicted == labels).sum().item()
                    epoch_total += labels.size(0)

                batch_count += 1

            except StopIteration:
                # Reset data loader
                multi_train_loader.reset()
                break
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        # Calculate epoch statistics
        avg_losses = {key: value / batch_count for key, value in epoch_losses.items()}
        train_acc = 100. * epoch_correct / epoch_total if epoch_total > 0 else 0.0

        # Update learning rates
        for scheduler in schedulers.values():
            scheduler.step()

        # Validation phase
        val_acc = evaluate_fusion_model(model, val_loaders, device, fusion_method, domain_adaptation)
        val_acc_percent = 100. * val_acc

        # Print epoch summary
        print(f"\nEpoch {epoch+1:3d}/{max_epochs} Summary:")
        print(f"  Train Loss: {avg_losses.get('total_loss', 0.0):.4f} | Train Acc: {train_acc:.2f}%")
        if 'adaptation_loss' in avg_losses:
            print(f"  Adaptation Loss: {avg_losses['adaptation_loss']:.4f}")
        if 'domain_loss' in avg_losses:
            print(f"  Domain Loss: {avg_losses['domain_loss']:.4f}")
        print(f"  Val Acc: {val_acc_percent:.2f}%")

        # Early stopping
        if early_stopping(val_acc, model, es_state, patience=EARLY_STOPPING_PATIENCE):
            print(f"Early stopping triggered! No improvement for {EARLY_STOPPING_PATIENCE} epochs")
            break

        print(f"  {'-'*50}")

    print(f"\n{'='*60}")
    print("Fusion Training Complete!")
    print(f"{'='*60}")

    # Load best model and evaluate on test set
    if 'best_model' in es_state and es_state['best_model'] is not None:
        model.load_state_dict(es_state['best_model'])

    return evaluate_fusion_model(model, test_loaders, device, fusion_method, domain_adaptation)


def evaluate_fusion_model(model, test_loaders: Dict, device, fusion_method: str = 'none',
                         domain_adaptation: str = 'none'):
    """
    Evaluate fusion model on test data

    Args:
        model: Model to evaluate
        test_loaders: Dictionary of test data loaders
        device: Evaluation device
        fusion_method: Fusion method being used
        domain_adaptation: Domain adaptation method

    Returns:
        Average test accuracy across all domains
    """
    model.eval()
    domain_accuracies = {}

    with torch.no_grad():
        for domain_name, test_loader in test_loaders.items():
            domain_correct = 0
            domain_total = 0

            for batch_data in test_loader:
                if len(batch_data) == 3:
                    data, labels, _ = batch_data
                else:
                    data, labels = batch_data

                data, labels = data.to(device), labels.to(device)
                data = normalize_data(data)

                # Forward pass
                if hasattr(model, 'forward') and 'dataset_name' in model.forward.__code__.co_varnames:
                    outputs = model(data, dataset_name=domain_name)
                elif domain_adaptation == 'ms_mda':
                    outputs = model(data, domain=domain_name)
                else:
                    outputs = model(data)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Take task predictions

                _, predicted = outputs.max(1)
                domain_correct += (predicted == labels).sum().item()
                domain_total += labels.size(0)

            if domain_total > 0:
                domain_accuracy = domain_correct / domain_total
                domain_accuracies[domain_name] = domain_accuracy
                print(f"Domain {domain_name} accuracy: {domain_accuracy:.4f}")

    # Return average accuracy across domains
    if domain_accuracies:
        avg_accuracy = sum(domain_accuracies.values()) / len(domain_accuracies)
        print(f"Average accuracy across domains: {avg_accuracy:.4f}")
        return avg_accuracy
    else:
        return 0.0
