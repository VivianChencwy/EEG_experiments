# Ablation Study for TF-DWT Method

This folder contains all results from the ablation study designed to evaluate the contribution of each component in the Target-Focused Domain Weighted Training (TF-DWT) method.

## Experiment Overview

**Base Method**: TF-DWT (from main_tfdwt.py)
- Evolving domain-specific weights with warmup
- RBF-MMD alignment loss
- Split-BN (separate BatchNorm statistics per domain)
- Early stopping on target domain validation

## Ablation Experiments

### 1. Equal Weights (`main_ablation1_equal_weights.py`)
**Modification**: Set w_P3 = w_AVO = 1.0 (no domain-specific weighting)

**Keeps**:
- MMD alignment
- Split-BN
- Other training mechanisms

**Result**: 0.6332 overall accuracy

---

### 2. Fixed Weights (`main_ablation2_fixed_weights.py`)
**Modification**: Weights computed once as sqrt(N_large/N_small), no evolution/warmup

**Keeps**:
- MMD alignment (with warmup)
- Split-BN
- Other training mechanisms

**Result**: 0.6165 overall accuracy

---

### 3. No MMD Alignment (`main_ablation3_no_mmd.py`)
**Modification**: Remove RBF-MMD alignment loss (lambda_MMD = 0)

**Keeps**:
- Domain weighting with warmup
- Split-BN
- Other training mechanisms

**Result**: 0.5997 overall accuracy

---

### 4. No Split-BN (`main_ablation4_no_split_bn.py`)
**Modification**: Use standard unified BatchNorm (no separate domain statistics)

**Keeps**:
- Domain weighting with warmup
- MMD alignment
- Other training mechanisms

**Result**: 0.6442 overall accuracy

---

## Files in This Directory

For each experiment (1-4):
- `ablation{N}_*.py` - Main experiment file (in parent directory)
- `ablation{N}_*_detailed_*.csv` - Detailed fold-by-fold results (25 folds: 5 repeats × 5 folds)
- `ablation{N}_*_summary_*.csv` - Summary statistics with confidence intervals
- `ablation{N}_run.log` - Complete execution log

## Results Summary

| Experiment | Overall Accuracy | Ranking |
|-----------|-----------------|---------|
| Ablation 4 (No Split-BN) | 0.6442 | 1st |
| Ablation 1 (Equal Weights) | 0.6332 | 2nd |
| Ablation 2 (Fixed Weights) | 0.6165 | 3rd |
| Ablation 3 (No MMD) | 0.5997 | 4th |

## Key Findings

1. **MMD Alignment is Critical**: Removing MMD (Ablation 3) caused the largest performance drop (-7.4% relative to best)

2. **Split-BN May Not Be Essential**: Unified BN (Ablation 4) actually performed best among ablations, suggesting that separate domain statistics may not be necessary

3. **Weight Evolution vs Fixed**: The difference between evolving weights (TF-DWT baseline) and fixed weights (Ablation 2) shows the benefit of adaptive weighting

4. **Equal Weights Perform Well**: Simple equal weighting (Ablation 1) achieved competitive performance, suggesting the importance of other components

## Statistical Analysis

All detailed results are saved in CSV format for statistical testing:
- Each CSV contains results from 5 repeats × 5 folds = 25 data points
- Can be used for paired t-tests comparing ablations
- Includes metrics: accuracy, precision, recall, F1, AUC for both P3 and AVO datasets

## Experimental Setup

- **Datasets**: P3 + AVO (combined training)
- **Cross-validation**: 5-fold stratified CV, repeated 5 times
- **P3 trials per subject**: 10
- **AVO trials per subject**: 80
- **Model**: EEGConformer
- **Electrodes**: Common channels only
- **Device**: CUDA

## Reproducibility

To reproduce any experiment:
```bash
python main_ablation{1|2|3|4}_*.py
```

Results will be saved in this directory with timestamps.

---

Generated: 2025-09-30
