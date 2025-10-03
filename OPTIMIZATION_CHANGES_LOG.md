# TF-DWT Optimization Changes Log

## Objective
Optimize TF-DWT (Target-Focused Domain Weighted Training) to achieve:
1. **AVO-focused scenario** (P3=80 trials/subject, AVO=10 trials/subject): AVO accuracy ≥ 0.66 (stable over 5 runs)
2. **P3-focused scenario** (P3=10 trials/subject, AVO=80 trials/subject): P3 accuracy ≥ 0.62 (stable over 5 runs)

## Baseline Performance
- Configuration: P3=80, AVO=10
- AVO Accuracy: 0.5834
- P3 Accuracy: 0.5902
- Overall: 0.5896
- **Status**: Below target

## Changes Implemented

### 1. Domain Weighting Enhancements (main_tfdwt.py)

#### Initial State
- Weight cap: 3.0
- Weighting formula: `sqrt(N_large / N_small)`
- MMD lambda: 0.1-0.3
- Warmup: 2-5 epochs

#### Round 1 Optimizations
```python
# Increased weight cap from 3.0 to 5.0
w_small_target = min(w_small_target, 5.0)

# Increased MMD alignment
lambda_mmd = 0.15 if ratio < 2.0 else (0.25 if ratio < 4.0 else 0.35)

# Extended warmup
warmup = max(3, min(8, int(0.15 * MAX_EPOCHS)))
```

#### Round 2 Optimizations
```python
# Further increased weight cap to 6.0
w_small_target = min(w_small_target, 6.0)

# Stronger MMD alignment
lambda_mmd = 0.2 if ratio < 2.0 else (0.3 if ratio < 4.0 else 0.4)

# Longer warmup
warmup = max(5, min(10, int(0.2 * MAX_EPOCHS)))
```

#### Round 3 Optimizations (Current)
```python
# Maximum weight cap: 8.0
# Changed from sqrt to linear scaling
w_small_target = max(1.0, max(1, n_large) / max(1, n_small))
w_small_target = min(w_small_target, 8.0)

# Maximum MMD alignment
lambda_mmd = 0.25 if ratio < 2.0 else (0.35 if ratio < 4.0 else 0.45)

# Extended warmup
warmup = max(6, min(12, int(0.25 * MAX_EPOCHS)))
```

### 2. Guard Mechanism Improvements (main_tfdwt.py)

#### Initial State
- Triggered after 3 consecutive drops
- Aggressive reduction: 0.8x weight, 0.5x lambda

#### Current State
```python
# Less aggressive guards
# Trigger only after 4 consecutive SIGNIFICANT drops (>0.01)
if len(val_history_small) >= 4:
    drops = [val_history_small[i-1] - val_history_small[i] > 0.01 for i in range(-3, 0)]
    if all(drops):
        new_w = max(1.0, cur_w * 0.9)  # Less aggressive: 0.9x instead of 0.8x
        new_lambda = max(0.0, cur_lambda * 0.8)  # Less aggressive: 0.8x instead of 0.5x
```

### 3. Focal Loss Implementation (main_tfdwt.py)

Added focal loss for the few-shot (small) domain to focus on hard examples:

```python
def compute_focal_loss(scores, targets, gamma=2.0, alpha=0.25):
    """Focal loss for handling class imbalance and hard examples"""
    ce_loss = F.cross_entropy(scores, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

# Applied in training loop for small domain
loss_small = compute_focal_loss(scores_small, y_small, gamma=2.5, alpha=0.35)
```

### 4. Training Hyperparameters (config.py)

#### Initial State
```python
LEARNING_RATE = 0.01
BATCH_SIZE = 128
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.25
MAX_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 50
```

#### Current State
```python
LEARNING_RATE = 0.008  # Reduced for better convergence
BATCH_SIZE = 96  # Smaller for better gradient estimates
WEIGHT_DECAY = 2e-4  # Stronger regularization
DROPOUT_RATE = 0.2  # Reduced for better learning
MAX_EPOCHS = 600  # More epochs for convergence
EARLY_STOPPING_PATIENCE = 70  # Increased patience
```

## Automation Scripts Created

1. **optimize_tfdwt.py**: Progressive hyperparameter search with multiple configurations
2. **quick_optimize.py**: Targeted optimization for specific scenarios
3. **auto_optimize_overnight.py**: Fully automated overnight optimization
4. **monitor_and_run_experiments.py**: Monitor and run experiments sequentially
5. **run_5_experiments.sh**: Bash script to run 5 sequential experiments
6. **check_status.sh**: Real-time status monitoring
7. **check_optimization_progress.sh**: Detailed progress tracking

## Testing Strategy

### Phase 1: AVO-Focused (P3=80, AVO=10)
- Run 5+ experiments with optimized settings
- Check if AVO accuracy ≥ 0.66 for 5 consecutive runs
- If successful, save configuration

### Phase 2: P3-Focused (P3=10, AVO=80)
- Apply same/similar optimizations
- Run 5+ experiments
- Check if P3 accuracy ≥ 0.62 for 5 consecutive runs
- If successful, save configuration

## Key Insights

1. **Domain Imbalance Handling**: The key challenge is that with P3=80 and AVO=10, there's a significant data imbalance (8:1 ratio). The model tends to optimize for the larger domain.

2. **Weighting Strategy**: Progressive increase in weight cap (3→5→6→8) and shift from sqrt to linear scaling provides stronger emphasis on the few-shot domain.

3. **Focal Loss**: Helps the model focus on hard examples in the few-shot domain, potentially improving accuracy on challenging cases.

4. **MMD Alignment**: Stronger alignment (lambda up to 0.45) helps transfer knowledge from large to small domain.

5. **Guard Mechanisms**: Less aggressive guards prevent premature reduction of beneficial strong weighting.

## Next Steps

1. **Monitor current experiments**: Automated scripts are running
2. **Analyze results**: Check if focal loss + linear weighting achieves target
3. **If unsuccessful**: Consider:
   - Meta-learning approaches (MAML, Prototypical Networks)
   - Data augmentation specific to small domain
   - Transfer learning with pre-training on large domain
   - Ensemble methods
   - Class rebalancing techniques

## Files Modified

1. `main_tfdwt.py`: Core TF-DWT algorithm with focal loss and enhanced weighting
2. `config.py`: Training hyperparameters
3. Multiple automation and monitoring scripts

## Optimization Rounds Summary

### Round 1: Focal Loss + Moderate Weighting
- Weight cap: 8.0 (linear)
- MMD lambda: 0.25-0.45
- Focal loss: gamma=2.5, alpha=0.35
- Batch size: 96, LR: 0.008
- **Result**: AVO=0.6186 (improvement +0.0352 from baseline, but still below 0.66 target)

### Round 2: Ultra-Aggressive (Current)
- Weight cap: 12.0 with 1.5x multiplier
- MMD lambda: 0.35-0.6
- Focal loss: gamma=3.0, alpha=0.5
- Batch size: 64, LR: 0.005
- Max epochs: 800, Patience: 100
- Dropout: 0.15, Weight decay: 3e-4
- **Status**: Running 5 experiments to test stability

## Current Status

- **Baseline Performance**: AVO=0.5834 (target: 0.66)
- **Round 1 Performance**: AVO=0.6186 (improvement: +0.0352)
- **Round 2**: Ultra-aggressive settings - 5 experiments in progress
- **Expected Completion**: ~1.5 hours per experiment × 5 = 7.5 hours total
- **Start Time**: 2025-10-02 00:49
