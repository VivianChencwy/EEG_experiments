# P3场景V5优化策略

## V4失败原因分析

### 观察到的问题
```
Epoch 1-2:  Val(P3) = 0.676 → 0.706  ✓ (峰值)
Epoch 3+:   Val(P3) = 0.471-0.618     (持续下降)
训练损失: 0.94 → 1.05 (上升)
训练准确率: 0.55 → 0.69 (上升)
```

### 根本原因
**Warmup过程导致的过拟合**
- 10个P3样本太少
- Warmup 40 epochs太长，权重从1.0→8.0的过程中模型过度优化AVO
- 早期(epoch 2)的良好状态被后续训练破坏

## V5策略：减轻Warmup

### 核心思路
- **保留warmup机制**，但大幅缩短和减弱
- 让模型更快稳定在最优参数，避免长时间调整

### 参数设计

#### 1. Warmup缩短 (从40→10 epochs)
```python
# V4: warmup = 40 epochs
# V5: warmup = 10 epochs (-75%)
warmup = max(5, min(10, int(0.1 * MAX_EPOCHS)))
```

#### 2. 降低目标权重 (减少25%)
```python
# V4: w_small = 8.1x
# V5: w_small = 6.0x (-25%)
w_small = float(np.clip(np.sqrt(max(ratio_ab, 1.0/ratio_ab)) * 2.2, 1.0, 10.0))
```

#### 3. 保持MMD和Proto温和
```python
# V4: lambda_mmd = 0.39, lambda_proto = 0.77
# V5: lambda_mmd = 0.30, lambda_proto = 0.60 (-20%)
lambda_mmd = 0.2 if overall_ratio < 2.0 else (0.25 if overall_ratio < 4.0 else 0.30)
lambda_proto = 0.4 if overall_ratio < 4.0 else 0.60
```

#### 4. 缩短patience (更早停止)
```python
# V4: patience = 150
# V5: patience = 50 (-67%)
# 原因: 避免在过拟合阶段浪费时间
```

#### 5. 其他参数保持V2
```python
dropout = 0.1
mixup_alpha = 0.30  # 回到V2
focal_gamma = 1.8   # 回到V2
```

### 完整参数对比

| 参数 | V2 | V4 | V5 | 理由 |
|------|----|----|-------|------|
| w_small | 7.9x | 8.1x | 6.0x | 降低过度优化 |
| warmup | 40 | 40 | **10** | 快速稳定 |
| λ_MMD | 0.35 | 0.39 | 0.30 | 温和对齐 |
| λ_proto | 0.7 | 0.77 | 0.60 | 温和引导 |
| Mixup α | 0.3 | 0.33 | 0.30 | 回归V2 |
| Focal γ | 1.8 | 2.0 | 1.8 | 回归V2 |
| Patience | 150 | 150 | **50** | 早停 |
| Dropout | 0.1 | 0.1 | 0.1 | 保持 |

## 预期效果

### Warmup时间轴对比
**V4 (40 epochs warmup):**
```
Epoch 1:  w=1.0 → P3好
Epoch 10: w=3.0 → P3开始下降
Epoch 20: w=5.0 → P3过拟合
Epoch 40: w=8.1 → P3严重过拟合
```

**V5 (10 epochs warmup):**
```
Epoch 1:  w=1.0 → P3好
Epoch 5:  w=3.5 → P3稳定
Epoch 10: w=6.0 → 达到目标，停止增长
Epoch 15-50: 保持w=6.0，等待收敛或early stop
```

### 成功标准
- P3验证准确率 ≥ 0.62
- 或至少稳定保持在0.60+ (不像V4那样持续下降)

## 风险评估

**低风险** ✓
- 仍保留warmup核心机制
- 只是缩短时长和降低强度
- 所有调整幅度可控(20-75%)
