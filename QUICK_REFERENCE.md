# TF-DWT 调参快速参考

## 🚀 一键运行命令

```bash
# 🔥 推荐流程 (总共2-4小时)
python run_ultrafast_test.py        # 10分钟验证
python run_tuning_example.py --mode quick  # 2-3小时调参

# ⚡ 最快验证 (10分钟)
python run_ultrafast_test.py

# 🎯 标准调参 (12-25小时)
python run_tuning_example.py --mode standard

# 🚄 GPU并行加速 (4-8小时，RTX 4090自动优化) ⭐ 最推荐
python parallel_gpu_tuning.py

# 🚄 CPU并行加速 (6-15小时，需要多核CPU)
python parallel_tuning.py --n_processes 4 --trials_per_process 25
```

## 📊 实时监控命令

```bash
# 查看状态
python check_status.py

# 监控训练日志
tail -f log_0909/TF_DWT_*.log

# 查看当前最佳结果
cat */tuning_results.json | grep best_score

# 查看进程
ps aux | grep tune_tfdwt
```

## 🎯 应用最佳参数

```bash
# 找到最佳配置并应用
cp */best_config.py config.py

# 验证结果
python main_tfdwt.py
```

## ⏱️ 时间预估

| 模式 | 命令 | 时间 | 试验数 |
|------|------|------|--------|
| 验证 | `python run_ultrafast_test.py` | 10分钟 | 1次简化 |
| 快速 | `python run_tuning_example.py --mode quick` | 2-3小时 | 5次试验 |
| 标准 | `python run_tuning_example.py --mode standard` | 12-25小时 | 50次试验 |
| **GPU并行** | `python parallel_gpu_tuning.py` | **4-8小时** | **75次试验** |
| CPU并行 | `python parallel_tuning.py --n_processes 4 --trials_per_process 25` | 6-15小时 | 100次试验 |

## 🛠️ 故障排除

| 问题 | 解决方法 |
|------|----------|
| "Could not extract accuracy" | 正常，系统会从日志文件中提取 |
| 进程卡住 | 检查 `tail -f log_0909/TF_DWT_*.log` |
| 想中断调参 | Ctrl+C，重新运行会继续 |
| 内存不足 | 在config.py中设置 `BATCH_SIZE = 16` |

## 📁 重要文件

- **结果**: `*/tuning_results.json`
- **最佳参数**: `*/best_config.py`
- **训练日志**: `log_0909/TF_DWT_*.log`
- **完整教程**: `TUNING_TUTORIAL.md`

## 🎯 目标准确率

- 基线: ~64%
- 快速调参: 66-70%
- 标准调参: 68-72%
- 大规模调参: 70-75%