# P3 V5改进版 - 解决Val-Test Gap问题

## 改动说明

### 问题诊断 (用户发现)
- **现象**: Validation P3准确率很高(0.62-0.79), 但Test只有0.57
- **原因**: Validation set太小(只有40 trials), Early stopping过拟合到validation set
- **Val-Test Gap**: 平均9.35个百分点 (不正常!)

### 核心修改

**config.py (lines 107-110):**
```python
# 之前 (V5原版)
TRAIN_SIZE = 0.7  # 280 trials
VAL_SIZE = 0.1    # 40 trials  ← 太小!
TEST_SIZE = 0.2   # 80 trials

# 现在 (V5改进版)
TRAIN_SIZE = 0.6  # 240 trials (-40)
VAL_SIZE = 0.2    # 80 trials (×2!)  ← 关键改动
TEST_SIZE = 0.2   # 80 trials (不变)
```

### 其他参数保持不变

**main_asmmd.py** V5参数全部保持:
- w_small: 6.0x
- lambda_mmd: 0.30
- lambda_proto: 0.60
- warmup: 10 epochs
- patience: 50
- mixup_alpha: 0.30
- focal_gamma: 1.8

## 预期效果

### 理论分析
1. **Validation更可靠**: 80 trials能更好代表真实分布
2. **Early stopping更准确**: 基于更大validation set选择checkpoint
3. **Val-Test gap缩小**: 预期从9%降到3-5%

### 成功标准
如果V5改进版成功:
- **Test准确率**: 应该接近之前的Validation准确率(0.65-0.70)
- **达标实验数**: 5次以上P3≥0.62 (目前只有4次)
- **平均P3准确率**: ≥0.62

### 对比基准
- V5原版 Val: 0.65-0.70 → Test: 0.5725 (gap=~10%)
- V5改进版预期: Val: 0.63-0.68 → Test: 0.60-0.65 (gap=~3%)

## 实验时间预估
- 训练数据减少: 280→240 trials (-14%)
- 每个epoch稍快,但总时长类似
- 预计: 15-20分钟完成25次实验

## 开始时间
2025-10-14 (待运行)
