# P3_80_AVO_10 最佳参数 - 20250929_172704
# 优化目标: AVO准确率 > 0.62
# 最佳配置: AVO_ULTRA
# 结果: AVO准确率 = 0.6395
# 达标: 是

P3_80_AVO_10_BEST_PARAMS = {'w_small_cap': 18.0, 'mmd_thresholds': (0.18, 0.45, 0.002, 0.012, 0.035), 'guard_factors': (0.01, 0.03), 'warmup_config': {'warmup_epochs': 38, 'warmup_lr_scale': 0.14, 'warmup_weight_scale': 0.24}, 'learning_rate': 0.03, 'batch_size': 5}

# 性能指标
PERFORMANCE = {
    'avo_accuracy': 0.6395,
    'p3_accuracy': 0.5474,
    'overall_accuracy': 0.6287,
    'result_file': 'tfdwt_detailed_results_20250929_172224.csv'
}

# 所有测试结果（按AVO准确率排序）
# 1. AVO_ULTRA: AVO=0.6395, P3=0.5474
# 2. AVO_AGGRESSIVE: AVO=0.6362, P3=0.5742
# 3. AVO_MODERATE: AVO=0.6351, P3=0.5625
# 4. AVO_BALANCED_PLUS: AVO=0.6313, P3=0.6128
# 5. AVO_EXTREME: AVO=0.6187, P3=0.5630
# 6. AVO_STRONG: AVO=0.6137, P3=0.5764
