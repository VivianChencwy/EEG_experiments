"""
Advanced model architectures for enhanced EEG classification
Includes attention-enhanced ShallowFBCSPNet and transformer-based models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from braindecode.models import ShallowFBCSPNet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from config import (
    INPUT_WINDOW_SAMPLES, use_subject_layer, EARLY_STOPPING_PATIENCE,
    LEARNING_RATE, WEIGHT_DECAY, GAMMA, MAX_EPOCHS, N_CLASSES,
    USE_DATA_AUGMENTATION, NOISE_STD, TIME_SHIFT_RANGE, LABEL_SMOOTHING, DROPOUT_RATE
)


class ChannelAttention(nn.Module):
    """Channel Attention Module (CAM) from CBAM."""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, t = x.size()
        
        # Global average pooling and max pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply sigmoid
        out = avg_out + max_out
        attention = self.sigmoid(out).view(b, c, 1)
        
        return x * attention.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial Attention Module (SAM) from CBAM."""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv1(x_cat))
        
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)."""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class AttentionEnhancedShallowFBCSPNet(nn.Module):
    """ShallowFBCSPNet enhanced with CBAM attention mechanisms."""
    
    def __init__(self, n_chans, n_outputs, n_times, final_conv_length='auto', 
                 pool_mode='mean', split_first_layer=True, batch_norm=True,
                 batch_norm_alpha=0.1, drop_prob=0.5):
        super(AttentionEnhancedShallowFBCSPNet, self).__init__()
        
        # Initialize base ShallowFBCSPNet layers manually to add attention
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        
        # Temporal convolution
        self.temporal_conv = nn.Conv2d(1, 40, (1, 25), stride=1, bias=False)
        
        # Spatial convolution  
        self.spatial_conv = nn.Conv2d(40, 40, (n_chans, 1), stride=1, bias=False)
        
        # Batch normalization
        self.bn = nn.BatchNorm2d(40, momentum=batch_norm_alpha, affine=True, eps=1e-5)
        
        # Activation
        self.activation = nn.ELU()
        
        # Add CBAM attention after spatial conv
        self.attention = CBAM(40, reduction=8)
        
        # Pooling
        pool_class = nn.AvgPool2d if pool_mode == 'mean' else nn.MaxPool2d
        self.pool = pool_class(kernel_size=(1, 75), stride=(1, 15))
        
        # Dropout
        self.dropout = nn.Dropout(drop_prob)
        
        # Calculate final conv length
        if final_conv_length == 'auto':
            # Estimate final length after convolutions and pooling
            dummy_input = torch.zeros(1, 1, n_chans, n_times)
            dummy_out = self._forward_conv_layers(dummy_input)
            final_conv_length = dummy_out.shape[3]
        
        # Final classification layer
        self.final_conv = nn.Conv2d(40, n_outputs, (1, final_conv_length), bias=True)
        
        # Initialize weights
        self._initialize_weights()
    
    def _forward_conv_layers(self, x):
        """Forward pass through convolutional layers only."""
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = self.activation(x)
        
        # Reshape for attention (merge spatial dim into batch)
        b, c, h, w = x.shape
        x_reshaped = x.view(b, c, h * w)
        x_attended = self.attention(x_reshaped)
        x = x_attended.view(b, c, h, w)
        
        x = self.pool(x)
        x = self.dropout(x)
        return x
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Input shape: (batch, channels, time) -> (batch, 1, channels, time)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        
        x = self._forward_conv_layers(x)
        x = self.final_conv(x)
        
        # Squeeze spatial dimensions
        x = x.squeeze(3).squeeze(2)
        
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        output = self.w_o(context)
        return output


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class EEGTransformer(nn.Module):
    """Transformer-based EEG classification model inspired by EEGformer."""
    
    def __init__(self, n_chans, n_outputs, n_times, d_model=128, n_heads=8, 
                 n_layers=4, d_ff=512, dropout=0.1):
        super(EEGTransformer, self).__init__()
        
        self.n_chans = n_chans
        self.n_times = n_times
        self.d_model = d_model
        
        # 1D CNN for initial feature extraction
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(n_chans, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ELU()
        )
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(n_times, d_model)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Global average pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_outputs)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, seq_len, d_model):
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
        
    def forward(self, x):
        # Input shape: (batch, channels, time)
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = self.cnn_layers(x)  # (batch, d_model, time)
        
        # Transpose for transformer: (batch, time, d_model)
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)
        x = self.dropout(x)
        
        # Transformer layers
        for transformer in self.transformer_layers:
            x = transformer(x)
        
        # Global pooling and classification
        x = x.transpose(1, 2)  # (batch, d_model, time)
        x = self.global_pool(x).squeeze(2)  # (batch, d_model)
        x = self.classifier(x)  # (batch, n_outputs)
        
        return x


class HybridCNN_Transformer(nn.Module):
    """Hybrid CNN-Transformer model combining spatial CNN with temporal Transformer."""
    
    def __init__(self, n_chans, n_outputs, n_times, d_model=128, n_heads=8, 
                 n_transformer_layers=3, dropout=0.1):
        super(HybridCNN_Transformer, self).__init__()
        
        # Spatial CNN branch (similar to ShallowFBCSPNet)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), padding=(0, 12)),
            nn.Conv2d(40, 40, (n_chans, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            CBAM(40)
        )
        
        # Temporal processing
        self.temporal_pool = nn.AvgPool2d((1, 4), stride=(1, 2))
        
        # Calculate conv output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_chans, n_times)
            conv_out = self.spatial_conv(dummy_input)
            conv_out = self.temporal_pool(conv_out)
            self.conv_out_size = conv_out.shape[-1]
        
        # Project to transformer dimension
        self.cnn_to_transformer = nn.Linear(40, d_model)
        
        # Transformer for temporal modeling
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_transformer_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_outputs)
        )
        
    def forward(self, x):
        # Input: (batch, channels, time) -> (batch, 1, channels, time)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        
        # Spatial CNN processing
        x = self.spatial_conv(x)  # (batch, 40, 1, time')
        x = self.temporal_pool(x)
        
        # Reshape for transformer: (batch, time', features)
        x = x.squeeze(2).transpose(1, 2)  # (batch, time', 40)
        x = self.cnn_to_transformer(x)    # (batch, time', d_model)
        
        # Transformer processing
        for transformer in self.transformer_layers:
            x = transformer(x)
        
        # Classification
        x = x.transpose(1, 2)  # (batch, d_model, time')
        x = self.classifier(x)
        
        return x


def create_advanced_model(model_type, n_channels, n_subjects=None, enable_subject_layer=None):
    """
    Create advanced model architectures.
    
    Parameters
    ----------
    model_type : str
        Type of model: 'attention_shallow', 'transformer', 'hybrid', 'lda'
    n_channels : int
        Number of input channels
    n_subjects : int, optional
        Number of subjects (for subject-specific layers)
    enable_subject_layer : bool, optional
        Whether to enable subject-specific layers
        
    Returns
    -------
    model : torch.nn.Module or sklearn model
    """
    
    if model_type == 'lda':
        return LDA()
    
    elif model_type == 'attention_shallow':
        return AttentionEnhancedShallowFBCSPNet(
            n_chans=n_channels,
            n_outputs=N_CLASSES,
            n_times=INPUT_WINDOW_SAMPLES,
            final_conv_length='auto',
            drop_prob=DROPOUT_RATE
        )
    
    elif model_type == 'transformer':
        return EEGTransformer(
            n_chans=n_channels,
            n_outputs=N_CLASSES,
            n_times=INPUT_WINDOW_SAMPLES,
            d_model=128,
            n_heads=8,
            n_layers=4,
            dropout=DROPOUT_RATE
        )
    
    elif model_type == 'hybrid':
        return HybridCNN_Transformer(
            n_chans=n_channels,
            n_outputs=N_CLASSES,
            n_times=INPUT_WINDOW_SAMPLES,
            d_model=128,
            n_heads=8,
            n_transformer_layers=3,
            dropout=DROPOUT_RATE
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")