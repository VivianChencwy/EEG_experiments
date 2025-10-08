# AVO场景成功参数总结

## 场景配置
```python
NESTED_CV_TRIALS_PER_SUBJECT_P3 = 80   # 大数据集
NESTED_CV_TRIALS_PER_SUBJECT_AVO = 10  # 小数据集 (优化目标)
```

## 模型架构
- **基础模型**: EEGConformer
- **方法**: Prototype-based TF-DWT (Target-Focused Domain Weighted Training)

## 核心算法参数

### 1. Domain Weighting (域权重)
```python
# 计算公式
w_small = clip(sqrt(N_large / N_small) * 3.0, min=1.0, max=12.0)

# 对于 P3=80, AVO=10:
# ratio = 80/10 = 8
# w_small = clip(sqrt(8) * 3.0, 1.0, 12.0) = clip(8.485, 1.0, 12.0) = 8.485

# 实际使用:
# - w_AVO (小域) ≈ 8.3-8.5x
# - w_P3 (大域) = 1.0x
# - 使用warmup: w = 1.0 + alpha * (w_target - 1.0)
```

### 2. MMD Alignment (分布对齐)
```python
# 根据样本比例自适应
lambda_mmd = 0.2 if ratio < 2.0 else (0.3 if ratio < 4.0 else 0.4)

# 对于 ratio=8:
lambda_mmd = 0.4

# 实际训练中: 随warmup逐渐增加到0.4
```

### 3. Prototype Loss (原型损失)
```python
lambda_proto = 0.5 if ratio < 4.0 else 0.8

# 对于 ratio=8:
lambda_proto = 0.8

# 作用: 从大域学习判别性知识，迁移到小域
```

### 4. Focal Loss (焦点损失)
```python
# 小域使用Mixup + Focal Loss
gamma = 2.0      # 关注困难样本程度
alpha = 0.5      # 类别权重

# 计算:
loss_small = mixup_criterion(scores, y_a, y_b, lam, gamma=2.0, alpha=0.5)
```

### 5. Mixup数据增强
```python
alpha = 0.4  # Beta分布参数

# 作用: 扩充小域样本，平滑决策边界
# 10 samples → ~45 virtual samples
```

## 训练超参数

### 基础参数
```python
BATCH_SIZE = 32                      # 小batch适合few-shot
LEARNING_RATE = 0.003                # 低学习率精细调整
WEIGHT_DECAY = 5e-4                  # L2正则化
DROPOUT_RATE = 0.1                   # 轻度dropout
MAX_EPOCHS = 1000                    # 充分训练
EARLY_STOPPING_PATIENCE = 150        # 大patience确保收敛
```

### 优化器和调度器
```python
optimizer = torch.optim.Adamax(
    model.parameters(),
    lr=0.003,
    weight_decay=5e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=1000
)
```

### Warmup策略
```python
warmup_epochs = max(20, min(40, int(0.4 * MAX_EPOCHS)))
# 实际: ~40 epochs

# Warmup schedule:
alpha = min(1.0, epoch / warmup_epochs)
w_current = 1.0 + alpha * (w_target - 1.0)
lambda_current = alpha * lambda_target
```

## 损失函数组成

```python
# 总损失
total_loss = (
    w_large * loss_large +           # 大域损失 (CE)
    w_small * loss_small +           # 小域损失 (Mixup Focal)
    lambda_mmd * loss_align +        # MMD对齐损失
    lambda_proto * loss_proto        # 原型损失
)

# 其中:
# - loss_large: 标准交叉熵
# - loss_small: Mixup + Focal Loss
# - loss_align: RBF-MMD
# - loss_proto: MSE to prototypes
```

## BatchNorm策略
```python
# 共享BN统计量 (非Split-BN)
# 小域借用全局BN的running_mean/var
# 原因: 10个样本无法准确估计统计量
```

## Guard机制
```python
# 防止过度优化导致崩溃
# 如果小域val连续4次显著下降(>0.01):
#   w_small *= 0.9
#   lambda_mmd *= 0.8

# 如果大域val连续4次显著下降:
#   lambda_mmd *= 0.85
```

## 早停策略
```python
# 基于小域验证性能
# Patience = 150 epochs
# 保存best model based on small domain val accuracy
```

## 关键设计决策

### 为什么这些参数有效?

1. **保守权重 (8.5x vs 15-17x)**
   - 避免过度优化小域破坏大域特征
   - 配合原型损失已足够

2. **降低MMD (0.4 vs 0.65-0.68)**
   - 过强MMD导致特征退化
   - 原型损失保证判别性，无需强MMD

3. **原型网络 (0.8)**
   - 核心创新
   - 从大域学习稳定的类原型
   - 小域向原型靠拢，保持判别性

4. **Mixup (0.4)**
   - 有效扩充10个样本
   - 平滑决策边界
   - 提升泛化能力

5. **小batch (32)**
   - 适合few-shot learning
   - 更好的梯度估计

6. **低学习率 (0.003)**
   - 精细调整小样本
   - 避免过拟合

7. **大patience (150)**
   - 确保充分收敛
   - 小样本需要更多epochs

## 电极配置
```python
electrode_list = 'all'  # 使用所有电极
# P3和AVO使用各自的全部电极
```

## 性能表现

### 成功率
- **7/9 = 77.8%** 达到 AVO ≥ 0.66

### 详细结果
| 实验 | AVO准确率 | 标准差 |
|------|-----------|--------|
| 最高 | 0.6708 | 0.0213 |
| 平均 | 0.6646 | ~0.025 |
| 最低 | 0.6603 | 0.0196 |

### 局限性
- 未实现连续5次达标的稳定性要求
- P3准确率偏低 (0.54-0.60)，说明对大域有一定损害

## 与旧方法对比

| 参数 | 旧方法 (Round 4-5) | 新方法 (Prototype) |
|------|-------------------|-------------------|
| 权重倍数 | 15-17x | 8.5x |
| MMD | 0.65-0.68 | 0.4 |
| Prototype | 无 | 0.8 |
| Focal gamma | 3.5-3.6 | 2.0 |
| Mixup | 无 | 0.4 |
| AVO平均 | 0.6607 | 0.6646 |
| 成功率 | 66.7% | 77.8% |

**改进**: +0.0039准确率, +11.1%成功率

## 文件对应关系

- **代码**: `main_avo.py` (原main_asmmd.py)
- **配置**: `config_avo.py` (原config.py)
- **结果**: `tfdwt_summary_stats_*.csv`
- **日志**: `TF_DWT_results_*.log`

## 下一步优化方向

1. **提高稳定性**:
   - 增加ensemble
   - 调整warmup策略
   - 优化early stopping

2. **保护大域性能**:
   - 调整权重比例
   - 增强P3特征学习

3. **P3场景**:
   - 当前参数是否适用于P3作为小域?
   - 可能需要场景特定的参数
