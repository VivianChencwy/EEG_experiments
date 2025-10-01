#!/usr/bin/env python3
"""
基于最佳参数(0.6301)进行精细调优
在成功配置附近进行小范围探索
"""

import subprocess
import sys
import time
from datetime import datetime
import pandas as pd
import glob

def get_latest_result():
    """获取最新结果文件的P3准确率"""
    files = sorted(glob.glob('tfdwt_detailed_results_*.csv'))
    if not files:
        return None, None

    latest_file = files[-1]
    df = pd.read_csv(latest_file)
    p3_acc = df['p3_accuracy'].mean()
    overall_acc = df['overall_accuracy'].mean()
    return latest_file, p3_acc, overall_acc

def run_experiment(params, name):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"🔬 {name}")
    print(f"{'='*60}")

    # 构建命令
    param_str = (
        f"--w_small_cap {params['w_small_cap']} "
        f"--mmd_alpha {params['mmd_thresholds'][0]} "
        f"--mmd_beta {params['mmd_thresholds'][1]} "
        f"--mmd_gamma {params['mmd_thresholds'][2]} "
        f"--mmd_delta {params['mmd_thresholds'][3]} "
        f"--mmd_epsilon {params['mmd_thresholds'][4]} "
        f"--guard_factor_1 {params['guard_factors'][0]} "
        f"--guard_factor_2 {params['guard_factors'][1]} "
        f"--warmup_epochs {params['warmup_config']['warmup_epochs']} "
        f"--warmup_lr_scale {params['warmup_config']['warmup_lr_scale']} "
        f"--warmup_weight_scale {params['warmup_config']['warmup_weight_scale']} "
        f"--learning_rate {params['learning_rate']} "
        f"--batch_size {params['batch_size']}"
    )

    cmd = f"python3 main_tfdwt.py combined P3_10 AVO_80 {param_str}"

    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    end_time = time.time()

    if result.returncode != 0:
        print(f"❌ 错误")
        return None

    # 获取结果
    result_file, p3_acc, overall_acc = get_latest_result()

    print(f"📊 P3: {p3_acc:.6f} | Overall: {overall_acc:.6f} | {(end_time-start_time)/60:.1f}分钟")

    return {
        'name': name,
        'params': params,
        'p3_accuracy': p3_acc,
        'overall_accuracy': overall_acc,
        'result_file': result_file
    }

def main():
    print("🎯 基于最佳参数(0.6301)进行精细调优")
    print("="*60)

    # 最佳基准参数 (达到0.6301)
    base_params = {
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

    # 在最佳参数附近微调 - 8组变体
    variants = [
        {
            'name': 'BASE_PLUS_WEIGHT',
            'params': {**base_params, 'w_small_cap': 13.0}
        },
        {
            'name': 'BASE_PLUS_LR',
            'params': {**base_params, 'learning_rate': 0.027}
        },
        {
            'name': 'BASE_TIGHTER_MMD',
            'params': {
                **base_params,
                'mmd_thresholds': (0.25, 0.55, 0.004, 0.018, 0.045)
            }
        },
        {
            'name': 'BASE_SMALLER_BATCH',
            'params': {**base_params, 'batch_size': 6}
        },
        {
            'name': 'BASE_LONGER_WARMUP',
            'params': {
                **base_params,
                'warmup_config': {
                    'warmup_epochs': 35,
                    'warmup_lr_scale': 0.18,
                    'warmup_weight_scale': 0.28
                }
            }
        },
        {
            'name': 'BASE_COMBO_1',
            'params': {
                **base_params,
                'w_small_cap': 13.5,
                'learning_rate': 0.027,
                'batch_size': 7
            }
        },
        {
            'name': 'BASE_COMBO_2',
            'params': {
                **base_params,
                'w_small_cap': 12.5,
                'mmd_thresholds': (0.28, 0.58, 0.0045, 0.019, 0.048),
                'learning_rate': 0.026
            }
        },
        {
            'name': 'BASE_OPTIMIZED',
            'params': {
                'w_small_cap': 13.0,
                'mmd_thresholds': (0.27, 0.57, 0.0048, 0.019, 0.047),
                'guard_factors': (0.018, 0.048),
                'warmup_config': {
                    'warmup_epochs': 32,
                    'warmup_lr_scale': 0.19,
                    'warmup_weight_scale': 0.29
                },
                'learning_rate': 0.026,
                'batch_size': 7
            }
        }
    ]

    results = []
    best_p3 = 0.6301

    for variant in variants:
        result = run_experiment(variant['params'], variant['name'])

        if result:
            results.append(result)
            if result['p3_accuracy'] > best_p3:
                best_p3 = result['p3_accuracy']
                print(f"🎉 新记录: {best_p3:.6f}")

    # 总结
    print(f"\n{'='*60}")
    print("📊 调优结果总结")
    print(f"{'='*60}")

    results_sorted = sorted(results, key=lambda x: x['p3_accuracy'], reverse=True)

    for i, r in enumerate(results_sorted, 1):
        marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{marker} {r['name']}: P3={r['p3_accuracy']:.6f} Overall={r['overall_accuracy']:.6f}")

    if results_sorted:
        best = results_sorted[0]
        print(f"\n🏆 最佳配置: {best['name']}")
        print(f"   P3 accuracy: {best['p3_accuracy']:.6f}")
        print(f"   Overall accuracy: {best['overall_accuracy']:.6f}")
        print(f"   结果文件: {best['result_file']}")

        # 保存最佳参数
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ultimate_p3_params_{timestamp}.py"

        with open(filename, "w") as f:
            f.write(f"# 精细调优最佳参数 - {timestamp}\n")
            f.write(f"# P3 accuracy: {best['p3_accuracy']:.6f}\n")
            f.write(f"# Overall accuracy: {best['overall_accuracy']:.6f}\n")
            f.write(f"# 配置: {best['name']}\n\n")
            f.write(f"ULTIMATE_P3_PARAMS = {best['params']}\n")

        print(f"\n✅ 参数已保存: {filename}")

if __name__ == "__main__":
    main()