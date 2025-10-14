# TF-DWT 超参数调优完整教程

本教程将指导您使用TF-DWT (Target-Focused Domain Weighted Training) 超参数调优系统，为您的EEG分类任务找到最佳参数组合。

## 🎯 系统概述

我们的调参系统针对您的配置进行优化：
- **P3数据**: 每个subject使用20个trials
- **AVO数据**: 每个subject使用200个trials
- **目标**: 在这种不平衡设置下最大化分类准确率

## 🚀 快速开始 (3步骤)

### 步骤1: 验证系统 (5-10分钟)
```bash
python run_ultrafast_test.py
```
这会运行一个快速测试来验证所有组件正常工作。

### 步骤2: 快速调参 (1.5-3小时)
```bash
python run_tuning_example.py --mode quick
```
运行5个试验的快速调参，获得初步的最佳参数。

### 步骤3: 完整调参 (可选，12-25小时)
```bash
python run_tuning_example.py --mode standard
```
运行50个试验的全面调参，获得更优的参数组合。

## 📋 所有可用选项

### 🎯 推荐的运行顺序

#### 1. 超快速测试 (⭐ 必须先运行)
```bash
python run_ultrafast_test.py
```
- **时间**: 5-10分钟
- **目的**: 验证系统功能
- **配置**: 简化设置 (30 epochs, 3 folds, 2 repeats)

#### 2. 快速调参 (⭐ 推荐)
```bash
# 方式A: 标准版本
python run_tuning_example.py --mode quick

# 方式B: 带实时进度监控版本
python run_quick_with_progress.py
```
- **时间**: 1.5-3小时
- **试验数**: 5个试验
- **配置**: 完整设置 (P3=20, AVO=200, 5 folds, 5 repeats)

#### 3. 标准调参
```bash
python run_tuning_example.py --mode standard
```
- **时间**: 12-25小时
- **试验数**: 50个试验
- **适合**: 有充足时间，追求更好结果

#### 4. 大规模调参
```bash
python run_tuning_example.py --mode extensive
```
- **时间**: 50-100小时
- **试验数**: 200个试验
- **适合**: 追求最优结果

#### 5. 并行调参 (⭐ 最高效)
```bash
# 4进程并行，总共100个试验
python parallel_tuning.py --n_processes 4 --trials_per_process 25

# 8进程并行 (需要8核以上CPU)
python parallel_tuning.py --n_processes 8 --trials_per_process 15
```
- **时间**: 6-15小时 (取决于CPU核数)
- **试验数**: 100-120个试验
- **适合**: 多核CPU，追求效率

### 🔍 网格搜索选项
```bash
python run_tuning_example.py --mode grid
```
- **时间**: 8-15小时
- **方法**: 系统性网格搜索
- **适合**: 精确验证特定参数范围

### 🎯 聚焦调参选项
```bash
python run_tuning_example.py --mode focused
```
- **时间**: 6-10小时
- **试验数**: 75个试验
- **方法**: 只调优最关键的参数
- **适合**: 时间有限但想获得好结果

## 📊 实时监控

### 方法1: 查看调参进度
```bash
# 查看当前最佳结果
cat quick_test_results/tuning_results.json

# 查看当前最佳参数配置
cat quick_test_results/best_config.py
```

### 方法2: 监控训练日志
```bash
# 查看最新的训练日志
tail -f log_0909/TF_DWT_*.log

# 查看最近的准确率
grep "Overall accuracy" log_0909/TF_DWT_*.log | tail -5
```

### 方法3: 监控系统资源
```bash
# 查看调参进程
ps aux | grep tune_tfdwt

# 查看GPU使用率 (如果有GPU)
nvidia-smi

# 查看CPU使用率
htop
```

## 🎯 参数调优空间

我们的系统会自动调优以下关键参数：

### 核心训练参数
- **LEARNING_RATE**: 0.0001 → 0.1 (对数分布)
- **WEIGHT_DECAY**: 1e-6 → 1e-2 (正则化强度)
- **DROPOUT_RATE**: 0.1 → 0.5
- **BATCH_SIZE**: [16, 24, 32, 48, 64]

### 模型架构
- **classifier**:
  - EEGConformer (默认推荐)
  - SepConv1DLite (轻量级)
  - EEGNetv4 (经典)
  - ShallowFBCSPNet (传统)
  - EEGChannelNet (注意力机制)

### TF-DWT特有参数
- **w_small_clip_max**: 2.0 → 8.0 (P3域权重上限)
- **lambda_mmd_base**: 0.05 → 0.5 (MMD对齐强度)
- **gradient_clip_norm**: 1.0 → 10.0 (梯度裁剪)
- **warmup_ratio**: 0.05 → 0.25 (预热比例)

### 数据增强参数
- **NOISE_STD**: 0.001 → 0.02 (噪声强度)
- **TIME_SHIFT_RANGE**: 2 → 15 (时间偏移范围)
- **LABEL_SMOOTHING**: 0.0 → 0.2 (标签平滑)

## 📈 预期结果

### 基线性能
- 默认参数: ~64% 准确率 (根据之前的测试)
- P3数据集: ~61% 准确率
- AVO数据集: ~65% 准确率

### 调参目标
- **快速调参 (5 trials)**: 66-70% 准确率
- **标准调参 (50 trials)**: 68-72% 准确率
- **大规模调参 (200 trials)**: 70-75% 准确率

### 关键改进点
1. **学习率优化**: +2-4% 准确率提升
2. **TF-DWT权重调整**: +3-5% 准确率提升
3. **模型架构选择**: +1-3% 准确率提升
4. **正则化优化**: +1-2% 准确率提升

## 🔧 故障排除

### 常见问题

#### 问题1: "Could not extract accuracy"
**解决**: 这是正常的，accuracy在log文件中，系统会自动找到。

#### 问题2: 进程看起来卡住了
**检查**:
```bash
# 查看是否在训练
tail -f log_0909/TF_DWT_*.log

# 查看进程是否存在
ps aux | grep main_tfdwt
```

#### 问题3: 内存不足
**解决**:
- 减少BATCH_SIZE (在config.py中设置为16或24)
- 使用更轻量级的模型 (SepConv1DLite)

#### 问题4: 想中断并恢复调参
**方法**:
- Ctrl+C 中断
- 重新运行相同命令，系统会自动从上次结果继续

## 📁 结果文件说明

调参完成后，您会在结果目录中找到：

### 主要结果文件
- **`tuning_results.json`**: 所有试验的完整结果
- **`best_config.py`**: 最佳参数的配置文件 (可直接使用)
- **`tuning_report.md`**: 详细分析报告
- **`tuning_*.log`**: 调参过程日志

### 如何使用最佳参数
```bash
# 1. 复制最佳配置
cp best_config.py config.py

# 2. 运行TF-DWT验证结果
python main_tfdwt.py

# 3. 或运行其他实验
python main.py
```

## 💡 最佳实践建议

### 1. 运行策略
1. **必须**: 先运行超快速测试验证系统
2. **推荐**: 运行快速调参获得初步最佳参数
3. **可选**: 如果时间充足，运行标准或大规模调参

### 2. 资源优化
- **单核**: 使用标准版本
- **多核 (4+)**: 使用并行调参
- **GPU**: 系统会自动使用GPU加速
- **有限时间**: 使用聚焦调参

### 3. 监控建议
- 定期检查 `tail -f log_0909/TF_DWT_*.log`
- 每几小时查看一次当前最佳结果
- 注意磁盘空间 (日志文件可能较大)

### 4. 安全建议
- 运行前备份重要数据
- 定期保存中间结果
- 使用 `screen` 或 `tmux` 运行长时间任务

## 🎯 示例运行序列

### 方案A: 快速验证 (适合初次使用)
```bash
# Step 1: 验证系统 (10分钟)
python run_ultrafast_test.py

# Step 2: 如果成功，运行快速调参 (3小时)
python run_tuning_example.py --mode quick

# Step 3: 使用最佳参数
cp quick_test_results/best_config.py config.py
python main_tfdwt.py
```

### 方案B: 追求最佳结果
```bash
# Step 1: 验证
python run_ultrafast_test.py

# Step 2: 并行大规模调参 (10-15小时)
python parallel_tuning.py --n_processes 4 --trials_per_process 50

# Step 3: 应用结果
cp parallel_tuning_results/best_config_parallel.py config.py
```

### 方案C: 时间有限
```bash
# Step 1: 验证
python run_ultrafast_test.py

# Step 2: 聚焦调参 (8小时)
python run_tuning_example.py --mode focused

# Step 3: 应用结果
cp focused_tuning_results/best_config.py config.py
```

## 🆘 获取帮助

如果遇到问题：
1. 检查本教程的故障排除部分
2. 查看日志文件了解详细错误信息
3. 确保所有依赖库已正确安装
4. 验证数据路径配置正确

现在选择一个方案开始您的调参之旅吧！🚀