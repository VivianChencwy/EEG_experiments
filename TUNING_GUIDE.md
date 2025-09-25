# TFDWT 完整调参系统使用指南

## 🎯 系统概述

本系统实现了高效的超参数调优策略，包含以下核心功能：

1. **分层调参** - 智能多阶段调参策略
2. **并行调参** - 多进程并行调参
3. **增量调参** - 基于历史结果继续优化
4. **实时监控** - 提供实时进度监控
5. **完整指南** - 一键运行所有功能

## 🚀 快速开始

### 1. 环境准备
```bash
conda activate eeg_realtime
cd /home/vivian/eeg/EEG_experiments
```

### 2. 系统检查
```bash
python complete_tuning_guide.py --check
```

### 3. 快速验证（推荐首次运行）
```bash
python quick_validation_test.py
```

## 📋 调参策略详解

### 1. 分层调参策略（推荐）
**特点**: 智能多阶段调参，效率最高
**时间**: 2-4小时
**命令**: 
```bash
python layered_tuning.py
```

**阶段说明**:
- **快速筛选**: 50 epochs, 2 folds, 20 trials - 快速筛选最有希望的参数
- **精细调优**: 150 epochs, 3 folds, 10 trials - 对筛选参数进行精细调优  
- **最终验证**: 300 epochs, 5 folds, 3 trials - 最终验证最佳参数

### 2. 并行调参策略
**特点**: 多进程并行，充分利用计算资源
**时间**: 1-2小时
**命令**:
```bash
python parallel_tuning.py --trials 20 --workers 4
```

### 3. 增量调参策略
**特点**: 基于历史结果继续优化，避免重复工作
**时间**: 30-60分钟
**命令**:
```bash
python incremental_tuning.py --trials 10 --strategy exploit_explore
```

### 4. 一键完整调参
**特点**: 集成所有功能，自动运行
**命令**:
```bash
python run_complete_tuning.py --strategy layered
```

## 📊 实时监控

### 快速状态检查
```bash
python monitor_tuning.py --mode quick
```

### 全面监控
```bash
python monitor_tuning.py --mode full
```

### 手动监控命令
```bash
# 查看最新日志
tail -f log_0909/TF_DWT_*.log

# 查看进程状态
ps aux | grep python | grep -E '(tune|main_tfdwt)'

# 查看GPU使用情况
nvidia-smi -l 1
```

## 🔧 参数说明

### 关键超参数范围
- **LEARNING_RATE**: 0.001 - 0.05 (学习率)
- **BATCH_SIZE**: 16, 24, 32, 48, 64 (批次大小)
- **DROPOUT_RATE**: 0.1 - 0.4 (Dropout率)
- **WEIGHT_DECAY**: 1e-6 - 1e-2 (权重衰减)
- **classifier**: EEGConformer, EEGNetv4, SepConv1DLite, ShallowFBCSPNet
- **NOISE_STD**: 0.001 - 0.02 (数据增强噪声)
- **TIME_SHIFT_RANGE**: 2 - 15 (时间偏移范围)
- **LABEL_SMOOTHING**: 0.0 - 0.2 (标签平滑)

## 📁 结果文件

### 调参结果目录
- `layered_tuning_results/` - 分层调参结果
- `parallel_tuning_results/` - 并行调参结果
- `incremental_tuning_results/` - 增量调参结果
- `quick_test_results/` - 快速测试结果

### 结果文件类型
- `*_results.json` - 详细调参结果
- `best_config.py` - 最佳参数配置
- `final_report.md` - 最终报告
- `tfdwt_*results*.csv` - 详细结果CSV
- `log_0909/TF_DWT_*.log` - 训练日志

## 🎯 推荐运行流程

### 首次使用
1. **系统检查**: `python complete_tuning_guide.py --check`
2. **快速验证**: `python quick_validation_test.py`
3. **分层调参**: `python layered_tuning.py`
4. **结果分析**: 查看 `layered_tuning_results/final_report.md`

### 资源充足时
1. **并行调参**: `python parallel_tuning.py --trials 20 --workers 4`
2. **增量优化**: `python incremental_tuning.py --trials 10`

### 继续优化
1. **分析历史**: `python incremental_tuning.py --analyze`
2. **增量调参**: `python incremental_tuning.py --trials 10 --strategy exploit`

## ⚠️ 注意事项

### 系统要求
- **内存**: 至少8GB RAM
- **存储**: 至少10GB可用空间
- **GPU**: 可选，但会显著加速（检测到RTX 4090）

### 安全特性
- 所有调参脚本支持中断后继续
- 结果实时保存，不会丢失
- 使用Ctrl+C安全中断

### 故障排除
- 如果遇到sys变量错误，已修复
- 如果进程卡住，检查日志文件
- 如果内存不足，减少batch_size或workers数量

## 🎉 成功案例

### 验证测试结果
- **快速验证**: 准确率 0.6038，耗时 0.9分钟
- **系统状态**: 所有必要文件存在，CUDA可用
- **实时监控**: 正常工作，可实时查看进度

### 调参效率提升
- **传统方法**: 12-25小时
- **分层调参**: 2-4小时（效率提升6-12倍）
- **并行调参**: 1-2小时（效率提升12-25倍）

## 📞 技术支持

如果遇到问题，请检查：
1. 系统状态：`python complete_tuning_guide.py --check`
2. 最新日志：`tail -f log_0909/TF_DWT_*.log`
3. 进程状态：`ps aux | grep python`

---

**🎯 开始您的调参之旅！**
