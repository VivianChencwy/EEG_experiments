"""
Advanced data augmentation techniques for EEG classification
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import random


class EEGAugmentation:
    """Advanced data augmentation for EEG signals."""
    
    def __init__(self, 
                 noise_std=0.01,
                 time_shift_range=10,
                 magnitude_warp_sigma=0.2,
                 time_warp_sigma=0.2,
                 channel_dropout_prob=0.1,
                 temporal_cutout_size=10,
                 mixup_alpha=0.4,
                 use_mixup=True,
                 freq_shift_delta=2,
                 amplitude_scale_range=(0.8, 1.2)):
        
        self.noise_std = noise_std
        self.time_shift_range = time_shift_range
        self.magnitude_warp_sigma = magnitude_warp_sigma
        self.time_warp_sigma = time_warp_sigma
        self.channel_dropout_prob = channel_dropout_prob
        self.temporal_cutout_size = temporal_cutout_size
        self.mixup_alpha = mixup_alpha
        self.use_mixup = use_mixup
        self.freq_shift_delta = freq_shift_delta
        self.amplitude_scale_range = amplitude_scale_range

    def add_gaussian_noise(self, x):
        """Add Gaussian noise to the signal."""
        if self.noise_std <= 0:
            return x
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def time_shift(self, x):
        """Apply time shifting augmentation."""
        if self.time_shift_range <= 0:
            return x
        
        batch_size, n_channels, n_times = x.shape
        augmented_x = x.clone()
        
        for i in range(batch_size):
            shift = random.randint(-self.time_shift_range, self.time_shift_range)
            if shift != 0:
                if shift > 0:
                    augmented_x[i, :, shift:] = x[i, :, :-shift]
                    augmented_x[i, :, :shift] = x[i, :, -shift:]
                else:
                    augmented_x[i, :, :shift] = x[i, :, -shift:]
                    augmented_x[i, :, shift:] = x[i, :, :-shift]
        
        return augmented_x

    def magnitude_warping(self, x):
        """Apply magnitude warping augmentation."""
        if self.magnitude_warp_sigma <= 0:
            return x
        
        batch_size, n_channels, n_times = x.shape
        
        # Generate warping curve
        knot = np.random.normal(loc=1.0, scale=self.magnitude_warp_sigma, size=(batch_size, n_channels, 4))
        orig_steps = np.arange(n_times)
        warping_steps = np.linspace(0, n_times-1, num=4)
        
        augmented_x = x.clone()
        
        for i in range(batch_size):
            for j in range(n_channels):
                # Interpolate warping curve
                f = interp1d(warping_steps, knot[i, j], kind='cubic', 
                           bounds_error=False, fill_value='extrapolate')
                warping_curve = f(orig_steps)
                
                # Apply warping
                augmented_x[i, j, :] = x[i, j, :] * torch.tensor(warping_curve, 
                                                                dtype=x.dtype, device=x.device)
        
        return augmented_x

    def time_warping(self, x):
        """Apply time warping augmentation."""
        if self.time_warp_sigma <= 0:
            return x
        
        batch_size, n_channels, n_times = x.shape
        
        # Generate time warping
        orig_steps = np.arange(n_times)
        random_warps = np.random.normal(loc=1.0, scale=self.time_warp_sigma, size=(batch_size, 4))
        
        augmented_x = x.clone()
        
        for i in range(batch_size):
            warp_steps = np.linspace(0, n_times-1, num=4) * random_warps[i]
            warp_steps = np.clip(warp_steps, 0, n_times-1)
            warp_steps = np.sort(warp_steps)
            
            # Time warping interpolation
            for j in range(n_channels):
                f = interp1d(warp_steps, x[i, j, np.linspace(0, n_times-1, num=4).astype(int)],
                           kind='linear', bounds_error=False, fill_value='extrapolate')
                augmented_x[i, j, :] = torch.tensor(f(orig_steps), dtype=x.dtype, device=x.device)
        
        return augmented_x

    def channel_dropout(self, x):
        """Apply channel dropout augmentation."""
        if self.channel_dropout_prob <= 0:
            return x
        
        batch_size, n_channels, n_times = x.shape
        augmented_x = x.clone()
        
        for i in range(batch_size):
            # Randomly select channels to drop
            n_drop = int(n_channels * self.channel_dropout_prob)
            if n_drop > 0:
                drop_channels = np.random.choice(n_channels, size=n_drop, replace=False)
                augmented_x[i, drop_channels, :] = 0
        
        return augmented_x

    def temporal_cutout(self, x):
        """Apply temporal cutout augmentation."""
        if self.temporal_cutout_size <= 0:
            return x
        
        batch_size, n_channels, n_times = x.shape
        augmented_x = x.clone()
        
        for i in range(batch_size):
            # Random start position
            start_pos = random.randint(0, max(1, n_times - self.temporal_cutout_size))
            end_pos = min(start_pos + self.temporal_cutout_size, n_times)
            
            # Zero out the selected temporal region
            augmented_x[i, :, start_pos:end_pos] = 0
        
        return augmented_x

    def frequency_shift(self, x, fs=128):
        """Apply frequency domain shifting."""
        if self.freq_shift_delta <= 0:
            return x
        
        batch_size, n_channels, n_times = x.shape
        augmented_x = x.clone()
        
        for i in range(batch_size):
            for j in range(n_channels):
                # Convert to frequency domain
                fft_signal = torch.fft.fft(x[i, j, :])
                freqs = torch.fft.fftfreq(n_times, 1/fs)
                
                # Apply frequency shift
                freq_shift = random.uniform(-self.freq_shift_delta, self.freq_shift_delta)
                phase_shift = 2 * np.pi * freq_shift * torch.arange(n_times, dtype=torch.float32) / fs
                phase_shift = phase_shift.to(x.device)
                
                # Apply phase shift
                shifted_signal = fft_signal * torch.exp(1j * phase_shift)
                
                # Convert back to time domain
                augmented_x[i, j, :] = torch.fft.ifft(shifted_signal).real
        
        return augmented_x

    def amplitude_scaling(self, x):
        """Apply amplitude scaling augmentation."""
        if self.amplitude_scale_range[0] >= self.amplitude_scale_range[1]:
            return x
        
        batch_size, n_channels, n_times = x.shape
        augmented_x = x.clone()
        
        for i in range(batch_size):
            scale = random.uniform(self.amplitude_scale_range[0], self.amplitude_scale_range[1])
            augmented_x[i, :, :] = x[i, :, :] * scale
        
        return augmented_x

    def mixup(self, x, y, alpha=None):
        """Apply mixup augmentation."""
        if not self.use_mixup:
            return x, y
        
        if alpha is None:
            alpha = self.mixup_alpha
        
        batch_size = x.size(0)
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        # Random permutation
        index = torch.randperm(batch_size).to(x.device)
        
        # Mix inputs and targets
        mixed_x = lam * x + (1 - lam) * x[index, :]
        
        if y.ndim == 1:  # Single labels
            y_a, y_b = y, y[index]
            return mixed_x, (y_a, y_b, lam)
        else:  # One-hot encoded
            y_a, y_b = y, y[index]
            mixed_y = lam * y + (1 - lam) * y_b
            return mixed_x, mixed_y

    def cutmix(self, x, y, alpha=1.0):
        """Apply CutMix augmentation."""
        batch_size, n_channels, n_times = x.shape
        
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        # Random permutation
        rand_index = torch.randperm(batch_size).to(x.device)
        
        # Generate random cut region
        cut_len = int(n_times * (1 - lam))
        cut_start = np.random.randint(0, n_times - cut_len + 1)
        cut_end = cut_start + cut_len
        
        # Apply cut and mix
        mixed_x = x.clone()
        mixed_x[:, :, cut_start:cut_end] = x[rand_index, :, cut_start:cut_end]
        
        # Adjust lambda based on actual cut ratio
        lam = 1 - (cut_end - cut_start) / n_times
        
        if y.ndim == 1:  # Single labels
            y_a, y_b = y, y[rand_index]
            return mixed_x, (y_a, y_b, lam)
        else:  # One-hot encoded
            y_a, y_b = y, y[rand_index]
            mixed_y = lam * y + (1 - lam) * y_b
            return mixed_x, mixed_y

    def __call__(self, x, y=None, training=True, apply_mixup=True):
        """
        Apply augmentation pipeline.
        
        Parameters
        ----------
        x : torch.Tensor
            Input EEG data (batch, channels, time)
        y : torch.Tensor, optional
            Labels for mixup
        training : bool
            Whether in training mode
        apply_mixup : bool
            Whether to apply mixup/cutmix
            
        Returns
        -------
        augmented_x : torch.Tensor
            Augmented EEG data
        augmented_y : torch.Tensor or tuple
            Augmented labels (if y provided)
        """
        if not training:
            return (x, y) if y is not None else x
        
        # Apply individual augmentations
        augmented_x = x
        
        # Basic augmentations
        if random.random() < 0.5:
            augmented_x = self.add_gaussian_noise(augmented_x)
        
        if random.random() < 0.3:
            augmented_x = self.time_shift(augmented_x)
        
        if random.random() < 0.3:
            augmented_x = self.magnitude_warping(augmented_x)
        
        if random.random() < 0.2:
            augmented_x = self.time_warping(augmented_x)
        
        if random.random() < 0.2:
            augmented_x = self.channel_dropout(augmented_x)
        
        if random.random() < 0.2:
            augmented_x = self.temporal_cutout(augmented_x)
        
        if random.random() < 0.3:
            augmented_x = self.amplitude_scaling(augmented_x)
        
        if random.random() < 0.1:
            augmented_x = self.frequency_shift(augmented_x)
        
        # Apply mixing augmentations
        if y is not None and apply_mixup:
            if random.random() < 0.3:
                if random.random() < 0.5:
                    augmented_x, augmented_y = self.mixup(augmented_x, y)
                else:
                    augmented_x, augmented_y = self.cutmix(augmented_x, y)
                return augmented_x, augmented_y
        
        return (augmented_x, y) if y is not None else augmented_x


class AdvancedLoss(torch.nn.Module):
    """Advanced loss function supporting mixup and label smoothing."""
    
    def __init__(self, alpha=1.0, gamma=2.0, label_smoothing=0.1, 
                 use_focal=True, use_label_smoothing=True):
        super(AdvancedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.use_focal = use_focal
        self.use_label_smoothing = use_label_smoothing
        
    def focal_loss(self, pred, target):
        """Compute focal loss."""
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
    
    def label_smoothing_loss(self, pred, target):
        """Compute label smoothing loss."""
        n_classes = pred.size(-1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.label_smoothing) + self.label_smoothing / n_classes
        return -(smooth_one_hot * F.log_softmax(pred, dim=1)).sum(dim=1).mean()
    
    def mixup_loss(self, pred, target_tuple):
        """Compute mixup loss."""
        y_a, y_b, lam = target_tuple
        loss_a = self.forward(pred, y_a, is_mixup=False)
        loss_b = self.forward(pred, y_b, is_mixup=False)
        return lam * loss_a + (1 - lam) * loss_b
    
    def forward(self, pred, target, is_mixup=None):
        """Forward pass."""
        # Handle mixup case
        if isinstance(target, tuple):
            return self.mixup_loss(pred, target)
        
        # Regular loss computation
        if self.use_focal:
            loss = self.focal_loss(pred, target)
        elif self.use_label_smoothing:
            loss = self.label_smoothing_loss(pred, target)
        else:
            loss = F.cross_entropy(pred, target)
        
        return loss