# 复现成功参数 - 20250929_142522
# 最佳P3 accuracy: 0.608796
# 平均P3 accuracy: 0.575692
# 5次运行稳定性验证

VERIFIED_P3_PARAMS = {'w_small_cap': 12.0, 'mmd_thresholds': (0.3, 0.6, 0.005, 0.02, 0.05), 'guard_factors': (0.02, 0.05), 'warmup_config': {'warmup_epochs': 30, 'warmup_lr_scale': 0.2, 'warmup_weight_scale': 0.3}, 'learning_rate': 0.025, 'batch_size': 8}

# 所有运行结果:
# Run 1: P3=0.596524, Overall=0.640522
# Run 2: P3=0.578385, Overall=0.626551
# Run 3: P3=0.572559, Overall=0.589043
# Run 4: P3=0.608796, Overall=0.635188
# Run 5: P3=0.522198, Overall=0.563884
