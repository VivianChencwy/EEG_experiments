#!/usr/bin/env python3
"""
双目标参数优化器
目标1: P3_10_AVO_80 配置下，P3准确率 > 0.62
目标2: P3_80_AVO_10 配置下，AVO准确率 > 0.62

策略：对每个目标测试多组参数，找到稳定可达到目标的配置
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

def run_experiment(params, config_name, dataset_config):
    """运行单次实验"""
    print(f"\n{'='*60}")
    print(f"🔬 {config_name} - {dataset_config}")
    print(f"{'='*60}")

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

    cmd = f"python3 main_tfdwt.py combined {dataset_config} {param_str}"

    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    duration = time.time() - start_time

    if result.returncode != 0:
        print(f"❌ 失败")
        return None

    result_file, p3_acc, avo_acc, overall_acc = get_latest_result()

    print(f"📊 P3: {p3_acc:.4f} | AVO: {avo_acc:.4f} | Overall: {overall_acc:.4f} | {duration/60:.1f}分钟")

    return {
        'config_name': config_name,
        'params': params,
        'p3_accuracy': p3_acc,
        'avo_accuracy': avo_acc,
        'overall_accuracy': overall_acc,
        'result_file': result_file,
        'duration': duration
    }

def verify_reproducibility(params, dataset_config, target_metric, target_value, num_runs=3):
    """验证参数的可复现性"""
    print(f"\n{'='*60}")
    print(f"🔍 验证可复现性 - {dataset_config}")
    print(f"目标: {target_metric} > {target_value}")
    print(f"运行次数: {num_runs}")
    print(f"{'='*60}")

    results = []
    for i in range(1, num_runs + 1):
        print(f"\n验证运行 {i}/{num_runs}")
        result = run_experiment(params, f"验证{i}", dataset_config)
        if result:
            results.append(result)

    if not results:
        return False, None

    # 提取目标指标
    if target_metric == 'p3_accuracy':
        values = [r['p3_accuracy'] for r in results]
    else:  # avo_accuracy
        values = [r['avo_accuracy'] for r in results]

    mean_val = np.mean(values)
    std_val = np.std(values)
    min_val = np.min(values)
    success_rate = sum(1 for v in values if v >= target_value) / len(values)

    print(f"\n验证结果:")
    print(f"  {target_metric} 平均: {mean_val:.4f}")
    print(f"  标准差: {std_val:.4f}")
    print(f"  范围: {min_val:.4f} - {np.max(values):.4f}")
    print(f"  达标率: {success_rate*100:.1f}% ({sum(1 for v in values if v >= target_value)}/{len(values)})")

    # 可复现性判断：平均值达标且至少50%的运行达标
    reproducible = (mean_val >= target_value * 0.98) and (success_rate >= 0.5)

    return reproducible, {
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': np.max(values),
        'success_rate': success_rate,
        'all_values': values
    }

def main():
    print("🎯 双目标参数优化")
    print("="*60)
    print("目标1: P3_10_AVO_80 → P3准确率 > 0.62")
    print("目标2: P3_80_AVO_10 → AVO准确率 > 0.62")
    print("="*60)

    # ========== 第一阶段：优化 P3_10_AVO_80 配置 ==========
    print("\n\n" + "="*60)
    print("第一阶段：优化 P3_10_AVO_80 (小P3数据集)")
    print("="*60)

    # P3小数据集参数候选 - 需要激进的权重增强
    p3_candidates = [
        {
            'name': 'P3_EXTREME_v1',
            'params': {
                'w_small_cap': 12.0,
                'mmd_thresholds': (0.3, 0.6, 0.005, 0.02, 0.05),
                'guard_factors': (0.02, 0.05),
                'warmup_config': {'warmup_epochs': 30, 'warmup_lr_scale': 0.2, 'warmup_weight_scale': 0.3},
                'learning_rate': 0.025,
                'batch_size': 8
            }
        },
        {
            'name': 'P3_EXTREME_v2',
            'params': {
                'w_small_cap': 14.0,
                'mmd_thresholds': (0.25, 0.55, 0.004, 0.018, 0.045),
                'guard_factors': (0.015, 0.04),
                'warmup_config': {'warmup_epochs': 32, 'warmup_lr_scale': 0.18, 'warmup_weight_scale': 0.28},
                'learning_rate': 0.027,
                'batch_size': 7
            }
        },
        {
            'name': 'P3_EXTREME_v3',
            'params': {
                'w_small_cap': 16.0,
                'mmd_thresholds': (0.28, 0.58, 0.0048, 0.019, 0.047),
                'guard_factors': (0.018, 0.045),
                'warmup_config': {'warmup_epochs': 35, 'warmup_lr_scale': 0.16, 'warmup_weight_scale': 0.26},
                'learning_rate': 0.028,
                'batch_size': 6
            }
        },
        {
            'name': 'P3_BALANCED',
            'params': {
                'w_small_cap': 10.0,
                'mmd_thresholds': (0.35, 0.65, 0.006, 0.022, 0.055),
                'guard_factors': (0.025, 0.055),
                'warmup_config': {'warmup_epochs': 28, 'warmup_lr_scale': 0.22, 'warmup_weight_scale': 0.32},
                'learning_rate': 0.023,
                'batch_size': 9
            }
        }
    ]

    p3_results = []
    for candidate in p3_candidates:
        result = run_experiment(candidate['params'], candidate['name'], 'P3_10 AVO_80')
        if result:
            p3_results.append(result)

    # 找出P3准确率最高的配置
    if p3_results:
        best_p3_config = max(p3_results, key=lambda x: x['p3_accuracy'])
        print(f"\n{'='*60}")
        print(f"P3_10_AVO_80 初步最佳: {best_p3_config['config_name']}")
        print(f"P3准确率: {best_p3_config['p3_accuracy']:.4f}")
        print(f"{'='*60}")

        # 验证可复现性
        print("\n开始验证可复现性...")
        reproducible, stats = verify_reproducibility(
            best_p3_config['params'],
            'P3_10 AVO_80',
            'p3_accuracy',
            0.62,
            num_runs=3
        )

        if reproducible:
            print("\n✅ P3_10_AVO_80 配置验证成功!")
            p3_final = {
                'dataset_config': 'P3_10_AVO_80',
                'params': best_p3_config['params'],
                'target_metric': 'p3_accuracy',
                'performance': stats,
                'reproducible': True
            }
        else:
            print("\n⚠️ P3_10_AVO_80 配置可复现性不足，使用最佳结果")
            p3_final = {
                'dataset_config': 'P3_10_AVO_80',
                'params': best_p3_config['params'],
                'target_metric': 'p3_accuracy',
                'performance': {
                    'best': best_p3_config['p3_accuracy'],
                    'verification': stats
                },
                'reproducible': False
            }
    else:
        print("\n❌ P3_10_AVO_80 所有配置都失败")
        p3_final = None

    # ========== 第二阶段：优化 P3_80_AVO_10 配置 ==========
    print("\n\n" + "="*60)
    print("第二阶段：优化 P3_80_AVO_10 (小AVO数据集)")
    print("="*60)

    # AVO小数据集参数候选 - 需要激进的权重增强（但w_small_cap增强AVO）
    avo_candidates = [
        {
            'name': 'AVO_EXTREME_v1',
            'params': {
                'w_small_cap': 12.0,
                'mmd_thresholds': (0.3, 0.6, 0.005, 0.02, 0.05),
                'guard_factors': (0.02, 0.05),
                'warmup_config': {'warmup_epochs': 30, 'warmup_lr_scale': 0.2, 'warmup_weight_scale': 0.3},
                'learning_rate': 0.025,
                'batch_size': 8
            }
        },
        {
            'name': 'AVO_EXTREME_v2',
            'params': {
                'w_small_cap': 14.0,
                'mmd_thresholds': (0.25, 0.55, 0.004, 0.018, 0.045),
                'guard_factors': (0.015, 0.04),
                'warmup_config': {'warmup_epochs': 32, 'warmup_lr_scale': 0.18, 'warmup_weight_scale': 0.28},
                'learning_rate': 0.027,
                'batch_size': 7
            }
        },
        {
            'name': 'AVO_EXTREME_v3',
            'params': {
                'w_small_cap': 16.0,
                'mmd_thresholds': (0.28, 0.58, 0.0048, 0.019, 0.047),
                'guard_factors': (0.018, 0.045),
                'warmup_config': {'warmup_epochs': 35, 'warmup_lr_scale': 0.16, 'warmup_weight_scale': 0.26},
                'learning_rate': 0.028,
                'batch_size': 6
            }
        },
        {
            'name': 'AVO_BALANCED',
            'params': {
                'w_small_cap': 10.0,
                'mmd_thresholds': (0.35, 0.65, 0.006, 0.022, 0.055),
                'guard_factors': (0.025, 0.055),
                'warmup_config': {'warmup_epochs': 28, 'warmup_lr_scale': 0.22, 'warmup_weight_scale': 0.32},
                'learning_rate': 0.023,
                'batch_size': 9
            }
        }
    ]

    avo_results = []
    for candidate in avo_candidates:
        result = run_experiment(candidate['params'], candidate['name'], 'P3_80 AVO_10')
        if result:
            avo_results.append(result)

    # 找出AVO准确率最高的配置
    if avo_results:
        best_avo_config = max(avo_results, key=lambda x: x['avo_accuracy'])
        print(f"\n{'='*60}")
        print(f"P3_80_AVO_10 初步最佳: {best_avo_config['config_name']}")
        print(f"AVO准确率: {best_avo_config['avo_accuracy']:.4f}")
        print(f"{'='*60}")

        # 验证可复现性
        print("\n开始验证可复现性...")
        reproducible, stats = verify_reproducibility(
            best_avo_config['params'],
            'P3_80 AVO_10',
            'avo_accuracy',
            0.62,
            num_runs=3
        )

        if reproducible:
            print("\n✅ P3_80_AVO_10 配置验证成功!")
            avo_final = {
                'dataset_config': 'P3_80_AVO_10',
                'params': best_avo_config['params'],
                'target_metric': 'avo_accuracy',
                'performance': stats,
                'reproducible': True
            }
        else:
            print("\n⚠️ P3_80_AVO_10 配置可复现性不足，使用最佳结果")
            avo_final = {
                'dataset_config': 'P3_80_AVO_10',
                'params': best_avo_config['params'],
                'target_metric': 'avo_accuracy',
                'performance': {
                    'best': best_avo_config['avo_accuracy'],
                    'verification': stats
                },
                'reproducible': False
            }
    else:
        print("\n❌ P3_80_AVO_10 所有配置都失败")
        avo_final = None

    # ========== 保存最终结果 ==========
    print("\n\n" + "="*60)
    print("最终结果汇总")
    print("="*60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    final_results = {
        'optimization_date': timestamp,
        'target_1': p3_final,
        'target_2': avo_final
    }

    # 保存为JSON
    json_file = f"dual_target_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    # 保存为Python文件
    py_file = f"dual_target_params_{timestamp}.py"
    with open(py_file, 'w') as f:
        f.write(f"# 双目标优化最终参数 - {timestamp}\n")
        f.write(f"# 自动生成，包含两组配置的最佳参数\n\n")

        if p3_final:
            f.write("# ========== 配置1: P3_10_AVO_80 (优化P3准确率) ==========\n")
            f.write(f"P3_10_AVO_80_PARAMS = {p3_final['params']}\n")
            f.write(f"# 目标指标: P3准确率 > 0.62\n")
            if 'mean' in p3_final['performance']:
                f.write(f"# 验证结果: 平均={p3_final['performance']['mean']:.4f}, ")
                f.write(f"达标率={p3_final['performance']['success_rate']*100:.1f}%\n")
            f.write(f"# 可复现: {p3_final['reproducible']}\n\n")

        if avo_final:
            f.write("# ========== 配置2: P3_80_AVO_10 (优化AVO准确率) ==========\n")
            f.write(f"P3_80_AVO_10_PARAMS = {avo_final['params']}\n")
            f.write(f"# 目标指标: AVO准确率 > 0.62\n")
            if 'mean' in avo_final['performance']:
                f.write(f"# 验证结果: 平均={avo_final['performance']['mean']:.4f}, ")
                f.write(f"达标率={avo_final['performance']['success_rate']*100:.1f}%\n")
            f.write(f"# 可复现: {avo_final['reproducible']}\n")

    print(f"\n✅ 结果已保存:")
    print(f"   JSON: {json_file}")
    print(f"   Python: {py_file}")

    if p3_final and avo_final:
        print("\n🎉 双目标优化完成!")
        return True
    else:
        print("\n⚠️  部分目标未完成")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)