# 极限优化设置 (Round 3)

## 启动时间
2025-10-02 上午（Round 2失败后）

## 问题分析
- Round 2最好AVO: 0.6323（目标0.66，差0.0377）
- 小样本域权重仍不够强
- MMD对齐仍不够强
- Focal loss仍有提升空间

## 极限设置（相比Round 2）

### main_tfdwt.py

#### 域权重
```python
# Round 2: 1.5x multiplier, cap 12.0
# Round 3: 2.0x multiplier, cap 20.0 ✓
w_small = float(np.clip(max(ratio_ab, 1.0/ratio_ab) * 2.0, 1.0, 20.0))
w_small_target = max(1.0, max(1, n_large) / max(1, n_small) * 2.0)
w_small_target = min(w_small_target, 20.0)
```
**提升**: 12x → 20x (提升67%)

#### MMD对齐强度
```python
# Round 2: 0.35-0.6
# Round 3: 0.5-0.8 ✓
lambda_mmd = 0.5 if ratio < 2.0 else (0.7 if ratio < 4.0 else 0.8)
```
**提升**: 最高0.6 → 0.8 (提升33%)

#### Focal Loss
```python
# Round 2: gamma=3.0, alpha=0.5
# Round 3: gamma=4.0, alpha=0.7 ✓
loss_small = compute_focal_loss(scores_small, y_small, gamma=4.0, alpha=0.7)
```
**提升**: gamma +33%, alpha +40%

#### Warmup
```python
# Round 2: max(10, min(20, int(0.3 * MAX_EPOCHS)))
# Round 3: max(15, min(30, int(0.35 * MAX_EPOCHS))) ✓
```
**提升**: 最多20 → 30 epochs

### config.py

#### Batch Size
```python
# Round 2: 64
# Round 3: 32 ✓
BATCH_SIZE = 32
```
**提升**: 减半，更精细的梯度

#### Learning Rate
```python
# Round 2: 0.005
# Round 3: 0.003 ✓
LEARNING_RATE = 0.003
```
**提升**: -40%，更精细学习

#### Max Epochs
```python
# Round 2: 800
# Round 3: 1000 ✓
MAX_EPOCHS = 1000
```
**提升**: +25%训练时间

#### Early Stopping Patience
```python
# Round 2: 100
# Round 3: 150 ✓
EARLY_STOPPING_PATIENCE = 150
```
**提升**: +50%耐心

#### Weight Decay
```python
# Round 2: 3e-4
# Round 3: 5e-4 ✓
WEIGHT_DECAY = 5e-4
```
**提升**: +67%正则化

#### Dropout
```python
# Round 2: 0.15
# Round 3: 0.1 ✓
DROPOUT_RATE = 0.1
```
**提升**: -33%，减少过度正则化

## 理论分析

### 为什么这些改动应该有效：

1. **20x权重**: 
   - AVO 10 trials vs P3 640 trials = 64:1比例
   - 20x权重可以给AVO损失约20倍的影响力
   - 理论上可以让AVO样本被重视64/20=3.2倍

2. **0.8 MMD对齐**:
   - 极强的域对齐，强制P3特征迁移到AVO
   - 风险：可能过度对齐导致P3性能下降
   - 但我们目标是AVO，可以接受

3. **Focal Loss gamma=4.0**:
   - 对于pt=0.9的简单样本：(1-0.9)^4 = 0.0001权重
   - 对于pt=0.5的困难样本：(1-0.5)^4 = 0.0625权重
   - 困难样本权重是简单样本的625倍！

4. **Batch Size 32**:
   - AVO训练集可能只有几十个样本
   - Batch 32确保每个batch有足够AVO样本
   - 更频繁的权重更新

5. **LR 0.003**:
   - 极小的学习率避免overfit小样本
   - 配合1000 epochs可以慢慢收敛

## 预期改进

基于Round 2: AVO=0.6323

极限设置预期提升：
- 权重增强：+0.015
- MMD增强：+0.010
- Focal loss增强：+0.010
- 小batch+低LR：+0.015

**预期总提升**: +0.05
**预期结果**: 0.6323 + 0.05 = 0.682 ✓ (超过0.66目标)

## 实验状态

AVO场景 (P3=80, AVO=10):
- Run 1: 进行中
- Run 2-5: 待运行

P3场景 (P3=10, AVO=80):
- 待AVO完成后运行

## 成功标准

- AVO场景：连续5次 ≥ 0.66
- P3场景：连续5次P3 ≥ 0.62

---
生成时间: 2025-10-02 上午
状态: 极限优化运行中
