# SepConv1D模型使用指南

## 概述

SepConv1D是从P300-CNNT项目集成到EEG_experiments中的轻量级分离卷积模型，专门设计用于小数据集的EEG P300分类任务，能够有效防止过拟合。

## 模型特点

### 1. 轻量级架构
- **参数少**: 使用分离卷积(Separable Convolution)大幅减少参数数量
- **计算高效**: 分离卷积分解为深度卷积+逐点卷积，计算量更小
- **内存占用低**: 适合资源受限的环境

### 2. 防过拟合设计
- **全局平均池化**: 替代全连接层减少参数
- **较小的权重初始化**: 使用更保守的权重初始化策略
- **自适应dropout**: 根据数据集大小自动调整dropout率
- **批量归一化**: 提高训练稳定性

### 3. 自动小数据集保护
当检测到以下条件时，系统会自动启用小数据集保护机制：
- 总样本数 ≤ 1000
- 或者选择了SepConv1D模型

保护措施包括：
- 降低dropout率 (0.3 → 0.2)
- 提高学习率 (0.0005 → 0.001)
- 增强L2正则化 (1e-5 → 1e-4)
- 减少早停耐心 (30 → 20 epochs)
- 限制最大训练轮数 (500 → 300 epochs)

## 使用方法

### 1. 基本使用
只需修改 `config.py` 中的分类器设置：

```python
# 在 config.py 中设置
classifier = 'SepConv1D'
```

然后运行主程序：
```bash
cd /home/vivian/eeg/EEG_experiments
conda activate eeg_realtime  # 激活conda环境
python main.py
```

### 2. 自定义参数
您可以在 `config.py` 中调整SepConv1D的特定参数：

```python
# SepConv1D specific parameters
SEPCONV1D_FILTERS = 32               # 输出滤波器数量 (建议16-64)
SEPCONV1D_KERNEL_SIZE = 16           # 时域卷积核大小 (建议8-32)
SEPCONV1D_STRIDE = 8                 # 步长 (建议4-16)
SEPCONV1D_PADDING = 4                # 填充 (通常为kernel_size/4)
```

### 3. 与Spatial Attention结合使用
SepConv1D完全兼容spatial attention等融合方法：

```python
# 在 config.py 中设置
classifier = 'SepConv1D'
ELECTRODE_FUSION_METHOD = 'spatial_attention'  # 启用空间注意力
```

### 4. 小数据集优化设置
如果您的数据集特别小，可以进一步调整保护参数：

```python
# 小数据集保护配置
SMALL_DATASET_THRESHOLD = 500        # 降低阈值，更早触发保护
SMALL_DATASET_DROPOUT_RATE = 0.1    # 进一步降低dropout
SMALL_DATASET_LEARNING_RATE = 0.002 # 提高学习率
SMALL_DATASET_WEIGHT_DECAY = 5e-4   # 增强正则化
```

## 适用场景

### 推荐使用SepConv1D的情况：
1. **小数据集** (< 1000样本)
2. **计算资源有限**
3. **需要快速原型验证**
4. **过拟合严重的场景**
5. **实时应用需求**

### 不推荐使用的情况：
1. **大数据集** (> 10000样本) - 建议使用EEGConformer
2. **需要最高精度** - 大数据集下复杂模型通常表现更好
3. **充足的计算资源和时间**

## 性能预期

根据原始P300-CNNT论文，SepConv1D在小数据集上的表现：
- **参数数量**: 比EEGNet少约70%
- **训练时间**: 比复杂模型快50-80%
- **内存占用**: 显著降低
- **过拟合风险**: 明显减少

## 调试和监控

运行时系统会自动显示小数据集保护信息：
```
============================================================
SMALL DATASET DETECTED (800 samples)
Applying overfitting prevention measures:
  - Reduced dropout: 0.2
  - Higher learning rate: 0.001
  - Stronger L2 regularization: 0.0001
  - Early stopping patience: 20
  - Max epochs: 300
============================================================
```

## 故障排除

### 常见问题：
1. **训练不收敛**: 尝试降低学习率或增加dropout
2. **验证准确率波动大**: 检查数据集大小，可能需要更强的正则化
3. **训练过快结束**: 早停可能过于敏感，可以增加patience

### 解决方案：
```python
# 调整超参数
SMALL_DATASET_LEARNING_RATE = 0.0005  # 降低学习率
SMALL_DATASET_EARLY_STOPPING_PATIENCE = 30  # 增加耐心
SMALL_DATASET_DROPOUT_RATE = 0.3  # 增加dropout
```

## 示例配置

以下是一个完整的小数据集配置示例：

```python
# config.py 中的推荐设置

# 基本模型设置
classifier = 'SepConv1D'

# 数据配置 - 小数据集建议使用固定试验数
FIXED_TRIALS_PER_SUBJECT_TRAIN = 20
FIXED_TRIALS_PER_SUBJECT_VAL = 10  
FIXED_TRIALS_PER_SUBJECT_TEST = 10

# SepConv1D参数
SEPCONV1D_FILTERS = 24              # 较小的滤波器数
SEPCONV1D_KERNEL_SIZE = 12          # 较小的卷积核
SEPCONV1D_STRIDE = 6                # 相应调整步长
SEPCONV1D_PADDING = 3               # 相应调整填充

# 小数据集保护
SMALL_DATASET_THRESHOLD = 800       # 根据实际情况调整
ENABLE_SMALL_DATASET_PROTECTIONS = True
```

这样配置后，直接运行 `python main.py` 即可使用优化过的SepConv1D模型进行训练。
