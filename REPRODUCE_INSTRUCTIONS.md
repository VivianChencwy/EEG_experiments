# 复现终极P3优化结果指南

## 目标
复现在极端不平衡条件下（P3=10, AVO=80）P3准确率达到0.6301的突破性结果

## 复现方法

### 方法1：使用自动化脚本（推荐）

```bash
python3 reproduce_ultimate_p3.py
```

这个脚本会：
1. 自动从备份文件加载成功的参数配置
2. 执行实验
3. 自动分析结果并与预期对比

### 方法2：手动执行命令

```bash
python3 main_tfdwt.py combined P3_10 AVO_80 \
  --w_small_cap 12.0 \
  --mmd_alpha 0.3 \
  --mmd_beta 0.6 \
  --mmd_gamma 0.005 \
  --mmd_delta 0.02 \
  --mmd_epsilon 0.05 \
  --guard_factor_1 0.02 \
  --guard_factor_2 0.05 \
  --warmup_epochs 30 \
  --warmup_lr_scale 0.2 \
  --warmup_weight_scale 0.3 \
  --learning_rate 0.025 \
  --batch_size 8
```

## 预期结果

- **P3准确率：** ~0.6301 (±0.02)
- **目标达成：** P3准确率 > 0.6 ✅
- **执行时间：** 30-60分钟
- **输出文件：** tfdwt_detailed_results_YYYYMMDD_HHMMSS.csv

## 验证结果

实验完成后，检查最新的结果文件：

```bash
# 查看最新结果文件
ls -la tfdwt_detailed_results_*.csv | tail -1

# 计算P3准确率
python3 -c "
import pandas as pd
import glob
files = sorted(glob.glob('tfdwt_detailed_results_*.csv'))
df = pd.read_csv(files[-1])
print(f'P3准确率: {df[\"p3_accuracy\"].mean():.4f}')
print(f'目标达成: {df[\"p3_accuracy\"].mean() > 0.6}')
"
```

## 参数配置说明

成功的关键参数配置：

- **w_small_cap=12.0:** 极高权重增强小数据集影响
- **mmd_thresholds=(0.3,0.6,0.005,0.02,0.05):** 超紧域对齐
- **guard_factors=(0.02,0.05):** 最小化保护机制
- **warmup策略:** 30轮激进预热
- **batch_size=8:** 小批次高频更新
- **learning_rate=0.025:** 较高学习率

## 备份文件

完整参数配置保存在：`optimal_tfdwt_parameters_backup.py`

## 注意事项

1. 确保config.py中设置：
   - NESTED_CV_TRIALS_PER_SUBJECT_P3 = 10
   - NESTED_CV_TRIALS_PER_SUBJECT_AVO = 80

2. 实验需要较长时间，建议在稳定环境中运行

3. 结果可能有±2%的随机变异，这是正常的CV变异