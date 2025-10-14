#!/usr/bin/env python3
"""
优化 P3_80_AVO_10 配置（小AVO数据集）
目标: AVO准确率 > 0.62
策略: 测试多组激进参数，增强小数据集（AVO）的影响
"""

import subprocess
import sys
import time
from datetime import datetime
import pandas as pd
import glob
import numpy as np
import json

def get_latest_result():
    """获取最新结果文件"""
    files = sorted(glob.glob('tfdwt_detailed_results_*.csv'))
    if files:
        df = pd.read_csv(files[-1])
        return files[-1], df['p3_accuracy'].mean(), df['avo_accuracy'].mean(), df['overall_accuracy'].mean()
    return None, None, None, None

def run_experiment(params, config_name):
    """运行单次实验"""
    print(f"\n{'='*60}")
    print(f"🔬 {config_name}")
    print(f"{'='*60}")

    # 显示关键参数
    print(f"w_small_cap: {params['w_small_cap']}")
    print(f"learning_rate: {params['learning_rate']}")
    print(f"batch_size: {params['batch_size']}")

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

    cmd = f"python3 main_tfdwt.py combined P3_80 AVO_10 {param_str}"

    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    duration = time.time() - start_time

    if result.returncode != 0:
        print(f"❌ 失败")
        return None

    result_file, p3_acc, avo_acc, overall_acc = get_latest_result()

    # 评估AVO准确率
    if avo_acc >= 0.65:
        marker = "🏆"
        status = "卓越"
    elif avo_acc >= 0.62:
        marker = "🎯"
        status = "达标"
    elif avo_acc >= 0.60:
        marker = "✅"
        status = "接近"
    else:
        marker = "⚠️"
        status = "偏低"

    print(f"{marker} AVO: {avo_acc:.4f} | P3: {p3_acc:.4f} | Overall: {overall_acc:.4f} | {status}")
    print(f"⏱️  {duration/60:.1f}分钟")

    return {
        'config_name': config_name,
        'params': params,
        'p3_accuracy': p3_acc,
        'avo_accuracy': avo_acc,
        'overall_accuracy': overall_acc,
        'result_file': result_file,
        'duration': duration
    }

def main():
    print("🎯 优化 P3_80_AVO_10 配置")
    print("="*60)
    print("数据集: P3=80, AVO=10 (小AVO数据集)")
    print("目标: AVO准确率 > 0.62")
    print("="*60)

    # AVO小数据集参数候选 - 6组不同强度的配置
    candidates = [
        {
            'name': 'AVO_MODERATE',
            'params': {
                'w_small_cap': 10.0,
                'mmd_thresholds': (0.35, 0.65, 0.006, 0.022, 0.055),
                'guard_factors': (0.025, 0.055),
                'warmup_config': {
                    'warmup_epochs': 28,
                    'warmup_lr_scale': 0.22,
                    'warmup_weight_scale': 0.32
                },
                'learning_rate': 0.023,
                'batch_size': 9
            }
        },
        {
            'name': 'AVO_STRONG',
            'params': {
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
        },
        {
            'name': 'AVO_AGGRESSIVE',
            'params': {
                'w_small_cap': 14.0,
                'mmd_thresholds': (0.25, 0.55, 0.004, 0.018, 0.045),
                'guard_factors': (0.015, 0.04),
                'warmup_config': {
                    'warmup_epochs': 32,
                    'warmup_lr_scale': 0.18,
                    'warmup_weight_scale': 0.28
                },
                'learning_rate': 0.027,
                'batch_size': 7
            }
        },
        {
            'name': 'AVO_EXTREME',
            'params': {
                'w_small_cap': 16.0,
                'mmd_thresholds': (0.2, 0.5, 0.003, 0.015, 0.04),
                'guard_factors': (0.012, 0.035),
                'warmup_config': {
                    'warmup_epochs': 35,
                    'warmup_lr_scale': 0.16,
                    'warmup_weight_scale': 0.26
                },
                'learning_rate': 0.028,
                'batch_size': 6
            }
        },
        {
            'name': 'AVO_ULTRA',
            'params': {
                'w_small_cap': 18.0,
                'mmd_thresholds': (0.18, 0.45, 0.002, 0.012, 0.035),
                'guard_factors': (0.01, 0.03),
                'warmup_config': {
                    'warmup_epochs': 38,
                    'warmup_lr_scale': 0.14,
                    'warmup_weight_scale': 0.24
                },
                'learning_rate': 0.03,
                'batch_size': 5
            }
        },
        {
            'name': 'AVO_BALANCED_PLUS',
            'params': {
                'w_small_cap': 13.0,
                'mmd_thresholds': (0.28, 0.58, 0.0048, 0.019, 0.047),
                'guard_factors': (0.018, 0.048),
                'warmup_config': {
                    'warmup_epochs': 33,
                    'warmup_lr_scale': 0.19,
                    'warmup_weight_scale': 0.29
                },
                'learning_rate': 0.026,
                'batch_size': 7
            }
        }
    ]

    results = []

    print(f"\n开始测试 {len(candidates)} 组参数配置...")

    for i, candidate in enumerate(candidates, 1):
        print(f"\n{'#'*60}")
        print(f"配置 {i}/{len(candidates)}")
        print(f"{'#'*60}")

        result = run_experiment(candidate['params'], candidate['name'])

        if result:
            results.append(result)
        else:
            print(f"⚠️  {candidate['name']} 运行失败")

    # 分析结果
    print(f"\n\n{'='*60}")
    print("📊 结果汇总")
    print(f"{'='*60}")

    if not results:
        print("❌ 所有配置都失败")
        return False

    # 按AVO准确率排序
    results_sorted = sorted(results, key=lambda x: x['avo_accuracy'], reverse=True)

    print(f"\n{'排名':<6} {'配置':<20} {'AVO准确率':<12} {'P3准确率':<12} {'总体准确率':<12}")
    print("-"*60)
    for i, r in enumerate(results_sorted, 1):
        marker = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        status = "✅" if r['avo_accuracy'] >= 0.62 else "  "
        print(f"{marker} {i:<4} {r['config_name']:<20} {r['avo_accuracy']:.4f} {status:<6} {r['p3_accuracy']:.4f}      {r['overall_accuracy']:.4f}")

    # 最佳配置
    best = results_sorted[0]
    print(f"\n{'='*60}")
    print(f"🏆 最佳配置: {best['config_name']}")
    print(f"{'='*60}")
    print(f"AVO准确率: {best['avo_accuracy']:.4f}")
    print(f"P3准确率: {best['p3_accuracy']:.4f}")
    print(f"总体准确率: {best['overall_accuracy']:.4f}")
    print(f"结果文件: {best['result_file']}")

    # 判断是否达标
    if best['avo_accuracy'] >= 0.62:
        print(f"\n🎉 达标! AVO准确率 {best['avo_accuracy']:.4f} ≥ 0.62")
        achieved_target = True
    else:
        print(f"\n⚠️  未达标，最佳AVO准确率为 {best['avo_accuracy']:.4f}")
        achieved_target = False

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存JSON
    json_file = f"avo_optimization_results_{timestamp}.json"
    json_data = {
        'optimization_date': timestamp,
        'dataset_config': 'P3_80_AVO_10',
        'target_metric': 'avo_accuracy',
        'target_value': 0.62,
        'achieved': achieved_target,
        'best_config': {
            'name': best['config_name'],
            'params': best['params'],
            'avo_accuracy': best['avo_accuracy'],
            'p3_accuracy': best['p3_accuracy'],
            'overall_accuracy': best['overall_accuracy'],
            'result_file': best['result_file']
        },
        'all_results': [
            {
                'name': r['config_name'],
                'avo_accuracy': r['avo_accuracy'],
                'p3_accuracy': r['p3_accuracy'],
                'overall_accuracy': r['overall_accuracy']
            }
            for r in results_sorted
        ]
    }

    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)

    # 保存Python文件
    py_file = f"avo_best_params_{timestamp}.py"
    with open(py_file, 'w') as f:
        f.write(f"# P3_80_AVO_10 最佳参数 - {timestamp}\n")
        f.write(f"# 优化目标: AVO准确率 > 0.62\n")
        f.write(f"# 最佳配置: {best['config_name']}\n")
        f.write(f"# 结果: AVO准确率 = {best['avo_accuracy']:.4f}\n")
        f.write(f"# 达标: {'是' if achieved_target else '否'}\n\n")

        f.write(f"P3_80_AVO_10_BEST_PARAMS = {best['params']}\n\n")

        f.write("# 性能指标\n")
        f.write(f"PERFORMANCE = {{\n")
        f.write(f"    'avo_accuracy': {best['avo_accuracy']:.4f},\n")
        f.write(f"    'p3_accuracy': {best['p3_accuracy']:.4f},\n")
        f.write(f"    'overall_accuracy': {best['overall_accuracy']:.4f},\n")
        f.write(f"    'result_file': '{best['result_file']}'\n")
        f.write(f"}}\n\n")

        f.write("# 所有测试结果（按AVO准确率排序）\n")
        for i, r in enumerate(results_sorted, 1):
            f.write(f"# {i}. {r['config_name']}: AVO={r['avo_accuracy']:.4f}, P3={r['p3_accuracy']:.4f}\n")

    print(f"\n✅ 结果已保存:")
    print(f"   JSON: {json_file}")
    print(f"   Python: {py_file}")

    return achieved_target

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)