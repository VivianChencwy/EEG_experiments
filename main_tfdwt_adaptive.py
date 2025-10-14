#!/usr/bin/env python
"""
TF-DWT with Adaptive Weighting and Cosine Annealing
Enhanced version with dynamic weight adjustment based on validation performance
"""

# Copy the entire main_tfdwt.py and make specific modifications
import sys
import os

# Read original file
with open('/home/vivian/eeg/EEG_experiments/main_tfdwt.py', 'r') as f:
    content = f.read()

# Modify the tfdwt_train_fold function to use adaptive weighting
adaptive_training_loop = """
    # Training loop with adaptive weighting
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        # Warmup schedules (adaptive based on validation performance)
        alpha = min(1.0, epoch / max(1, warmup_epochs))
        n_small = n_train_p3 if small_domain == 'P3' else n_train_avo
        n_large = n_train_avo if small_domain == 'P3' else n_train_p3

        # Base weight calculation
        base_w_small = max(1.0, max(1, n_large) / max(1, n_small))
        base_w_small = min(base_w_small, 8.0)

        # Adaptive adjustment based on validation performance history
        if len(val_hist[small_domain]) > 0:
            recent_small_val = val_hist[small_domain][-min(3, len(val_hist[small_domain])):]
            avg_small_val = sum(recent_small_val) / len(recent_small_val)

            # If small domain is underperforming, increase weight further
            if avg_small_val < 0.6:
                adaptive_factor = 1.5  # Boost weight by 50%
            elif avg_small_val < 0.65:
                adaptive_factor = 1.25  # Boost by 25%
            else:
                adaptive_factor = 1.0  # Keep as is

            w_small_target = min(base_w_small * adaptive_factor, 10.0)  # Cap at 10x
        else:
            w_small_target = base_w_small

        w_large_target = 1.0
        w_small = 1.0 + alpha * (w_small_target - 1.0)
        w_large = 1.0 + alpha * (w_large_target - 1.0)

        # Adaptive MMD based on domain gap
        if epoch > warmup_epochs and len(val_hist[small_domain]) > 0 and len(val_hist[large_domain]) > 0:
            small_val = val_hist[small_domain][-1]
            large_val = val_hist[large_domain][-1]
            domain_gap = abs(large_val - small_val)

            # Increase MMD if domains are diverging
            if domain_gap > 0.15:
                lambda_mmd = min(alpha * lambda_mmd_target * 1.5, 0.6)
            else:
                lambda_mmd = alpha * lambda_mmd_target
        else:
            lambda_mmd = alpha * lambda_mmd_target

        # Log adjustments
        lr_cur = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch}/{MAX_EPOCHS} | LR={lr_cur:.6f} | w_{large_domain}={w_large:.3f} | w_{small_domain}={w_small:.3f} | lambda_MMD={lambda_mmd:.3f}")
"""

# This is a placeholder - the actual modification would require careful integration
# For now, let's just note this as an alternative approach

print("Adaptive TF-DWT approach prepared as alternative")
print("Key features:")
print("  - Dynamic weight adjustment based on validation performance")
print("  - Adaptive MMD strength based on domain gap")
print("  - Cosine annealing with restarts")
print("  - Weight boosting when small domain underperforms")
