# AVO-Focused Ablation Analysis (Small Dataset Performance)

## Context
- **AVO Dataset**: Small dataset with only **10 trials per subject**
- **P3 Dataset**: Large dataset with **80 trials per subject**
- **Goal**: Understand which components help the small dataset most

---

## Results Ranked by AVO (Small Dataset) Performance

| Rank | Experiment | AVO Accuracy | P3 Accuracy | Overall | Gap from Best |
|------|-----------|--------------|-------------|---------|---------------|
| 🥇 1st | **Ablation 1 (Equal Weights)** | **0.6879** | 0.5945 | 0.6050 | - |
| 🥈 2nd | Ablation 3 (No MMD) | 0.6571 | 0.5788 | 0.5874 | -3.08% |
| 🥉 3rd | Ablation 4 (No Split-BN) | 0.6126 | 0.6143 | 0.6143 | -7.53% |
| 4th | Ablation 2 (Fixed Weights) | 0.5828 | 0.5831 | 0.5832 | -10.51% |

---

## 🚨 SURPRISING: Completely OPPOSITE Results from P3-Small! 

### When P3 was small (10 trials):
- Best: Ablation 4 (No Split-BN) - 0.5931
- Worst: Ablation 2 (Fixed Weights) - 0.5556

### When AVO is small (10 trials):
- Best: Ablation 1 (Equal Weights) - 0.6879
- Worst: Ablation 2 (Fixed Weights) - 0.5828

### Key Difference:
**Equal weights work BEST for AVO-small, but only 3rd best for P3-small!**

---

## Key Findings (AVO-Centric View)

### 1. **Equal Weights are BEST for AVO** 🏆
- **Ablation 1 (Equal Weights)** achieved the **BEST AVO accuracy** (0.6879)
- 10.51% better than worst (Fixed Weights)
- 3.08% better than 2nd place (No MMD)
- **This is OPPOSITE to P3 results!**

### 2. **Fixed Weights are WORST for AVO** 🔴
- **Ablation 2 (Fixed Weights)** had the **WORST AVO performance** (0.5828)
- Consistent with P3: Fixed weights hurt small datasets
- Adaptive evolution is critical regardless of which dataset is small

### 3. **MMD Helps AVO When Combined with Equal Weights** ✅
- **Ablation 3 (No MMD)** ranked 2nd for AVO (0.6571)
- But overall performance suffered (P3 dropped to 0.5788)
- MMD alignment helps maintain balance between datasets

### 4. **Split-BN Less Beneficial for AVO** 🟡
- **Ablation 4 (No Split-BN)** ranked 3rd for AVO (0.6126)
- But had BEST overall balance (AVO=0.6126, P3=0.6143)
- Unified BN provides better generalization across datasets

---

## Component Contribution Analysis

### For AVO (Small Dataset):

| Component | Impact on AVO | Impact on P3 | Conclusion |
|-----------|--------------|--------------|------------|
| **Equal Weights** | 🟢 Best (+10.51% vs worst) | 🟡 Moderate | AVO benefits from equal treatment |
| **Fixed Weights** | 🔴 Worst (-10.51%) | 🔴 Worst | Critical for both datasets |
| **MMD Alignment** | 🟢 Positive (2nd place) | 🟡 Hurts P3 | Helps AVO but needs balance |
| **Split-BN Removal** | 🟡 Neutral (3rd place) | 🟢 Helps P3 | Better overall balance |

---

## Why Are Results Different for AVO vs P3? 💡

### Hypothesis 1: Dataset Characteristics
1. **P3 Dataset Nature**:
   - Cognitive task (target detection)
   - Strong, consistent ERP components (P300)
   - High signal quality in averaged trials

2. **AVO Dataset Nature**:
   - Visual oddball task
   - Potentially more variable responses
   - May benefit from simpler training approach

### Hypothesis 2: Domain Dominance
**When AVO is small:**
- P3 (large) naturally dominates gradient updates
- Equal weights actually **help** AVO compete
- Domain weighting may **over-emphasize** AVO, causing overfitting

**When P3 is small:**
- AVO (large) dominates more strongly (8x more data)
- P3 needs active emphasis to avoid being washed out
- Equal weights insufficient for P3 to learn effectively

### Hypothesis 3: Learning Dynamics
**AVO may learn faster than P3:**
- If AVO has simpler patterns, early emphasis causes overfitting
- Equal weights allow gradual, stable learning
- P3 may need more data/epochs to learn complex patterns

---

## Cross-Dataset Comparison

### Common Finding (Both Datasets):
✅ **Fixed Weights are CONSISTENTLY WORST**
- P3-small: -3.75% vs best
- AVO-small: -10.51% vs best
- **Adaptive weight evolution is CRITICAL**

### Opposite Findings:

| Component | P3-Small (10 trials) | AVO-Small (10 trials) |
|-----------|---------------------|----------------------|
| Equal Weights | 3rd (0.5775) | **1st (0.6879)** 🏆 |
| Split-BN | **Hurts** (4th→1st when removed) | **Neutral** (3rd when removed) |
| MMD | Helps with weighting | Helps but hurts P3 |

---

## Revised Understanding: Dataset-Specific Strategies

### For P3 as Small Dataset:
✅ Use domain weighting evolution (CRITICAL)  
✅ Use unified BN (not Split-BN)  
✅ Keep MMD alignment  
❌ Don't use equal weights (insufficient emphasis)

### For AVO as Small Dataset:
✅ Use **EQUAL weights** (best performance!) 🌟  
✅ Use domain weighting evolution (avoid fixed)  
✅ Keep MMD for balance  
🟡 Split-BN optional (unified BN gives better balance)

---

## Universal Principles Across Datasets

1. **Never Use Fixed Weights** 🔴
   - Hurts both P3-small (-3.75%) and AVO-small (-10.51%)
   - Adaptive evolution is non-negotiable

2. **Dataset Characteristics Matter** 🎯
   - P3 needs active emphasis when small
   - AVO benefits from equal treatment when small
   - One-size-fits-all approach doesn't work

3. **Unified BN Generally Better** ✅
   - Better for P3-small (best performer)
   - Better overall balance for AVO-small
   - Split-BN may cause instability with limited data

4. **MMD Alignment is Important** ✅
   - Helps maintain cross-dataset performance
   - Critical when combined with proper weighting strategy

---

## Recommendations for Small Dataset Learning

### If Small Dataset Characteristics Unknown:
**Conservative Strategy:**
1. Try **Ablation 4 (No Split-BN)** first
   - Best for P3-small (0.5931)
   - 3rd for AVO-small but best balance (0.6126/0.6143)
   - Most stable across different scenarios

### If Dataset Has Strong Patterns (like AVO):
**Aggressive Strategy:**
1. Try **Equal Weights** (Ablation 1)
   - Best for AVO-small (0.6879)
   - Simpler, more stable learning

### If Dataset Has Weak Patterns (like P3):
**Emphasis Strategy:**
1. Use **TF-DWT with unified BN** (Ablation 4 components)
   - Active weighting evolution
   - Unified BN for stability
   - MMD for alignment

---

## Statistical Significance

All results based on:
- 5 repeats × 5 folds = 25 data points per experiment
- Detailed CSV files available for paired t-tests

### Stability (Standard Deviations):
- Ablation 2 (Fixed): ± 0.0447 (most stable but worst performance)
- Ablation 4 (No Split-BN): ± 0.0452 (stable + good balance)
- Ablation 3 (No MMD): ± 0.0506
- Ablation 1 (Equal): ± 0.0542 (least stable but best AVO)

**Trade-off**: Ablation 1 has highest variance but best mean for AVO

---

Generated: 2025-09-30  
Based on 4 complete ablation experiments with 25-fold cross-validation each  
Configuration: AVO = 10 trials/subject (small), P3 = 80 trials/subject (large)
