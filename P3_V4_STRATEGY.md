# P3场景V4优化策略

## 背景

### 所有实验结果总览
| 版本 | P3准确率 | 参数特点 | 结果 |
|------|----------|----------|------|
| 初始 | 0.5720 | 未知 | - |
| V1激进 | 0.5757 | w*3.5, MMD=0.5, proto=1.0 | 过强，轻微过拟合 |
| **V2保守** | **0.5876** | w*2.8, MMD=0.35, proto=0.7 | **当前最佳** ✓ |
| V3平衡 | 0.5789 | w*3.1, patience=200, dropout=0.15 | 训练太长，不稳定 |

**目标**: 0.62 (当前最佳差距: 0.0324 = 5.2%)

## V4策略：V2微调版

### 核心思路
- V2已经很接近目标，只需要**轻微增强**
- 避免V3的过度训练问题
- 添加标签平滑防止过拟合

### 参数设计

#### 1. Domain Weight (微调+5%)
```python
# V2: sqrt(ratio) * 2.8 → ~7.9x
# V4: sqrt(ratio) * 2.95 → ~8.3x  (+5%)
w_small = float(np.clip(np.sqrt(max(ratio_ab, 1.0/ratio_ab)) * 2.95, 1.0, 12.0))
```

#### 2. MMD Loss (微调+12%)
```python
# V2: 0.35
# V4: 0.39 (+12%)
lambda_mmd = 0.2 if overall_ratio < 2.0 else (0.30 if overall_ratio < 4.0 else 0.39)
```

#### 3. Prototype Loss (微调+10%)
```python
# V2: 0.7
# V4: 0.77 (+10%)
lambda_proto = 0.5 if overall_ratio < 4.0 else 0.77
```

#### 4. Mixup (微调+10%)
```python
# V2: alpha=0.3
# V4: alpha=0.33 (+10%)
```

#### 5. Focal Loss (微调+11%)
```python
# V2: gamma=1.8
# V4: gamma=2.0 (+11%)
```

#### 6. Label Smoothing (新增)
```python
# V2: 无
# V4: 0.05 (轻微平滑，防止过拟合)
```

#### 7. 训练参数 (保持V2)
```python
patience = 150  # 保持V2，不用V3的200
dropout = 0.1   # 保持V2，不用V3的0.15
warmup = 40     # 保持V2，不用V3的80
```

### 完整参数对比表

| 参数 | V2保守 | V4微调 | 变化 |
|------|--------|--------|------|
| w_small multiplier | 2.8→7.9x | 2.95→8.3x | +5% |
| λ_MMD | 0.35 | 0.39 | +12% |
| λ_proto | 0.7 | 0.77 | +10% |
| Mixup α | 0.3 | 0.33 | +10% |
| Focal γ | 1.8 | 2.0 | +11% |
| Label Smoothing | 0.0 | 0.05 | **新增** |
| Patience | 150 | 150 | 不变 |
| Dropout | 0.1 | 0.1 | 不变 |
| Warmup | 40 | 40 | 不变 |

### 预期效果

基于线性外推:
- V2基础: 0.5876
- 参数增强5-12%
- 标签平滑改善泛化
- **预期**: 0.60-0.62

### 风险评估

**低风险** ✓
- 所有调整幅度 ≤ 12%
- 保持V2稳定的训练配置
- 标签平滑是标准正则化手段

## 如果V4失败，V5备选方案

### 方案A: 学习率调度优化
- Cosine annealing with restarts
- 初始LR从0.003调整到0.002或0.005

### 方案B: 批量大小调整
- 从32调整到16 (更频繁更新)
- 或调整到64 (更稳定梯度)

### 方案C: 两阶段训练 (方法层面改进)
- Stage 1: 大域(AVO)预训练特征提取器
- Stage 2: 冻结底层，只微调顶层分类器

### 方案D: 改进Prototype计算
- 当前: EMA更新大域prototypes
- 改进: 添加momentum contrast机制

### 方案E: 增强数据增强
- 添加时间扭曲 (Time Warping)
- 添加频域增强 (Frequency Masking)
