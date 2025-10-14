# P3场景优化计划

## 当前状况

### 配置
- P3 trials/subject: 10 (小数据集)
- AVO trials/subject: 80 (大数据集)
- 目标: P3 ≥ 0.62

### 现有结果
- P3准确率: 0.6043 ± 0.0290
- 差距: 0.0157 (2.6%)
- 状态: **未达标**

## 与AVO场景对比

| 指标 | AVO场景 | P3场景 | 差异 |
|------|---------|--------|------|
| 小域样本 | AVO=10 | P3=10 | 相同 |
| 大域样本 | P3=80 | AVO=80 | 相同 |
| 小域准确率 | 0.6646 | 0.6043 | -0.0603 |
| 成功率 | 77.8% | 0% | -77.8% |

**关键发现**:
- 样本配置完全对称
- 但P3作为小域时性能明显更差
- 说明**P3数据本身更难**或需要**不同的优化策略**

## 可能原因

### 1. P3和AVO数据的本质差异

**P3任务特点**:
- Oddball paradigm
- 注意力依赖
- P300成分(300ms latency)
- 信号较弱，个体差异大

**AVO任务特点**:
- Audiovisual oddball
- 多模态融合
- 信号可能更强
- 个体差异可能较小

**推测**: P3的10 trials可能信息量不足，相比AVO的10 trials更难学习

### 2. 当前方法可能偏向AVO

检查代码中是否有隐含的AVO偏好:
- 早停基于哪个域?
- 数据增强是否适用P3?
- 原型计算是否适合P3?

## 优化策略

### 方案1: 调整参数针对P3 (优先)

**假设**: P3需要更强的优化力度

#### 参数调整:
```python
# 当前AVO成功参数
w_small = sqrt(ratio) * 3.0, cap at 12x  → 实际8.5x
lambda_mmd = 0.4
lambda_proto = 0.8
mixup_alpha = 0.4
focal_gamma = 2.0

# P3优化参数 (提案)
w_small = sqrt(ratio) * 3.5, cap at 14x  → 实际9.9x  (+16%)
lambda_mmd = 0.5                         → (+25%)
lambda_proto = 1.0                       → (+25%)
mixup_alpha = 0.5                        → (+25%)
focal_gamma = 2.5                        → (+25%)
```

**理由**: P3更难，需要稍强的优化，但不能太激进

### 方案2: 数据增强强化

**P3特定增强**:
- 增加Mixup强度 (0.4 → 0.5-0.6)
- 添加时间扭曲 (Time Warping)
- 添加噪声注入 (Gaussian noise)

**风险**: 可能破坏P300时序特性

### 方案3: 调整早停策略

**当前**: 基于小域(P3)验证性能

**可能问题**: P3波动大，容易early stop过早

**改进**:
- 增加patience (150 → 200)
- 使用moving average作为early stop判据
- 考虑多指标组合

### 方案4: 两阶段训练

**Stage 1**: 在大域(AVO=80)预训练
- 学习稳定的特征提取器
- 50-100 epochs

**Stage 2**: 冻结部分层，微调P3
- 只训练最后几层
- 专注优化P3分类

## 实施计划

### Step 1: 微调参数测试 (方案1)
时间: ~30分钟/次

修改`main_asmmd.py`:
```python
# get_symmetric_adjustments函数
w_small = float(np.clip(np.sqrt(max(ratio_ab, 1.0/ratio_ab)) * 3.5, 1.0, 14.0))
lambda_mmd = 0.2 if overall_ratio < 2.0 else (0.35 if overall_ratio < 4.0 else 0.5)
lambda_proto = 0.6 if overall_ratio < 4.0 else 1.0

# mixup和focal
mixup_alpha = 0.5
focal_gamma = 2.5
```

### Step 2: 如果成功
- 运行5次验证稳定性
- 保存P3成功参数

### Step 3: 如果失败
- 尝试方案3 (调整early stop)
- 或方案4 (两阶段训练)

## 预期结果

### 乐观估计
- P3达到 0.62-0.63
- 5次成功率 > 80%

### 保守估计
- P3达到 0.615-0.620
- 可能需要多次迭代

### 悲观情况
- P3仍然 < 0.62
- 需要改变方法(迁移学习/元学习)

## 风险评估

### 低风险调整
- ✓ 微调权重 (+16%)
- ✓ 微调MMD (+25%)
- ✓ 调整patience

### 中风险调整
- ⚠ Mixup增强
- ⚠ Focal loss gamma

### 高风险调整
- ⚠⚠ 两阶段训练 (架构改变)
- ⚠⚠ 数据增强 (可能破坏信号)

## 下一步行动

1. 实施方案1 (微调参数)
2. 运行1次测试
3. 评估结果
4. 如果接近目标，运行5次验证
5. 如果不足，迭代调整或换方案
