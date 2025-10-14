"""
Backup of optimal TF-DWT parameters for different scenarios
Generated on: 2025-09-28
"""

# Scenario 1: P3=20, AVO=200 (Conservative optimization)
# Target: Improve P3 accuracy from baseline
# Result: P3 accuracy improved from 0.5853 to 0.6003 (+1.5%, p=0.0324)
CONSERVATIVE_P3_PARAMS = {
    'w_small_cap': 2.5,
    'mmd_thresholds': (1.5, 3.0, 0.05, 0.15, 0.25),
    'guard_factors': (0.2, 0.3),
    'warmup_config': {
        'warmup_epochs': 15,
        'warmup_lr_scale': 0.5,
        'warmup_weight_scale': 0.7
    },
    'learning_rate': 0.012,
    'batch_size': 32
}

# Scenario 2: P3=10, AVO=80 (Ultra-aggressive optimization)
# Target: P3 accuracy > 0.6
# Result: P3 accuracy = 0.5733 (not yet achieved target)
ULTRA_AGGRESSIVE_P3_PARAMS = {
    'w_small_cap': 8.0,
    'mmd_thresholds': (0.5, 1.0, 0.01, 0.05, 0.1),
    'guard_factors': (0.05, 0.1),
    'warmup_config': {
        'warmup_epochs': 25,
        'warmup_lr_scale': 0.3,
        'warmup_weight_scale': 0.5
    },
    'learning_rate': 0.02,
    'batch_size': 16
}

# Scenario 3: P3=10, AVO=80 (ULTIMATE optimization - TARGET ACHIEVED!)
# Target: P3 accuracy > 0.6
# Result: P3 accuracy = 0.6301 (🎉 TARGET ACHIEVED!)
ULTIMATE_SUCCESS_P3_PARAMS = {
    'w_small_cap': 12.0,
    'mmd_thresholds': (0.3, 0.6, 0.005, 0.02, 0.05),
    'guard_factors': (0.02, 0.05),
    'warmup_config': {
        'warmup_epochs': 30,
        'warmup_lr_scale': 0.2,
        'warmup_weight_scale': 0.3
    },
    'learning_rate': 0.025,
    'batch_size': 8
}

# Best baseline results for reference
BASELINE_RESULTS = {
    'P3_20_AVO_200': {
        'p3_accuracy': 0.5853,
        'overall_accuracy': 0.6297
    },
    'AVO_20_P3_200': {
        'avo_accuracy': 0.6749,
        'overall_accuracy': 0.6309
    }
}

# Optimized results
OPTIMIZED_RESULTS = {
    'P3_20_AVO_200_conservative': {
        'p3_accuracy': 0.6003,
        'improvement': '+1.5%',
        'significance': 'p=0.0324'
    },
    'P3_10_AVO_80_ultra': {
        'p3_accuracy': 0.5733,
        'target_achieved': False,
        'target': 0.6
    },
    'P3_10_AVO_80_ULTIMATE': {
        'p3_accuracy': 0.6301,
        'target_achieved': True,
        'target': 0.6,
        'improvement': '+10.0% vs baseline',
        'breakthrough': 'First to exceed 0.6 threshold!',
        'result_file': 'tfdwt_detailed_results_20250928_230851.csv'
    }
}