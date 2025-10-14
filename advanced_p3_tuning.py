#!/usr/bin/env python3
"""
高级P3参数调优 - 系统探索参数空间以突破0.63
基于当前0.6008的结果，探索更极端的参数组合
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
    return latest_file, p3_acc

def run_experiment(params, name):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"🔬 实验: {name}")
    print(f"{'='*60}")

    for key, value in params.items():
        print(f"  {key}: {value}")

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

    print(f"\n⏱️  开始执行...")
    start_time = time.time()

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    end_time = time.time()
    duration = end_time - start_time

    print(f"✅ 执行完成 (用时: {duration/60:.1f}分钟)")

    if result.returncode != 0:
        print(f"❌ 错误: {result.stderr}")
        return None

    # 获取结果
    result_file, p3_acc = get_latest_result()

    if p3_acc is None:
        print("❌ 无法读取结果")
        return None

    print(f"📊 P3 accuracy: {p3_acc:.6f}")
    print(f"   目标 > 0.63: {'✅' if p3_acc > 0.63 else '❌'}")

    return {
        'name': name,
        'params': params,
        'p3_accuracy': p3_acc,
        'result_file': result_file,
        'duration': duration
    }

def main():
    print("🚀 高级P3参数调优")
    print("="*60)
    print("目标: P3 accuracy > 0.63")
    print("当前最佳: 0.6301")
    print("="*60)

    # 设计6组渐进式激进参数
    param_configs = [
        {
            'name': 'HYPER_BOOST_v1',
            'params': {
                'w_small_cap': 15.0,  # 更高权重
                'mmd_thresholds': (0.25, 0.5, 0.003, 0.015, 0.04),  # 更紧
                'guard_factors': (0.015, 0.04),  # 更小
                'warmup_config': {
                    'warmup_epochs': 35,
                    'warmup_lr_scale': 0.15,
                    'warmup_weight_scale': 0.25
                },
                'learning_rate': 0.028,
                'batch_size': 6
            }
        },
        {
            'name': 'HYPER_BOOST_v2',
            'params': {
                'w_small_cap': 18.0,  # 极高权重
                'mmd_thresholds': (0.2, 0.4, 0.002, 0.01, 0.03),
                'guard_factors': (0.01, 0.03),
                'warmup_config': {
                    'warmup_epochs': 40,
                    'warmup_lr_scale': 0.12,
                    'warmup_weight_scale': 0.2
                },
                'learning_rate': 0.03,
                'batch_size': 4
            }
        },
        {
            'name': 'PRECISION_MAX',
            'params': {
                'w_small_cap': 14.0,
                'mmd_thresholds': (0.15, 0.3, 0.001, 0.008, 0.025),  # 超紧对齐
                'guard_factors': (0.008, 0.025),
                'warmup_config': {
                    'warmup_epochs': 45,
                    'warmup_lr_scale': 0.1,
                    'warmup_weight_scale': 0.18
                },
                'learning_rate': 0.032,
                'batch_size': 4
            }
        },
        {
            'name': 'BALANCED_EXTREME',
            'params': {
                'w_small_cap': 16.0,
                'mmd_thresholds': (0.28, 0.55, 0.004, 0.018, 0.045),
                'guard_factors': (0.012, 0.035),
                'warmup_config': {
                    'warmup_epochs': 38,
                    'warmup_lr_scale': 0.14,
                    'warmup_weight_scale': 0.22
                },
                'learning_rate': 0.029,
                'batch_size': 5
            }
        },
        {
            'name': 'ULTRA_TIGHT',
            'params': {
                'w_small_cap': 13.0,
                'mmd_thresholds': (0.1, 0.25, 0.0008, 0.006, 0.02),  # 最紧对齐
                'guard_factors': (0.005, 0.02),
                'warmup_config': {
                    'warmup_epochs': 50,  # 超长预热
                    'warmup_lr_scale': 0.08,
                    'warmup_weight_scale': 0.15
                },
                'learning_rate': 0.035,
                'batch_size': 3
            }
        },
        {
            'name': 'MAXIMUM_POWER',
            'params': {
                'w_small_cap': 20.0,  # 最大权重
                'mmd_thresholds': (0.18, 0.35, 0.0015, 0.009, 0.028),
                'guard_factors': (0.006, 0.022),
                'warmup_config': {
                    'warmup_epochs': 42,
                    'warmup_lr_scale': 0.11,
                    'warmup_weight_scale': 0.19
                },
                'learning_rate': 0.033,
                'batch_size': 3
            }
        }
    ]

    results = []
    best_result = None
    best_p3_acc = 0.6301  # 已知最佳结果

    for config in param_configs:
        result = run_experiment(config['params'], config['name'])

        if result is None:
            print(f"⚠️  {config['name']} 执行失败，跳过")
            continue

        results.append(result)

        # 更新最佳结果
        if result['p3_accuracy'] > best_p3_acc:
            best_p3_acc = result['p3_accuracy']
            best_result = result
            print(f"\n🎉 新记录！P3 accuracy: {best_p3_acc:.6f}")

        # 如果达到0.65，可以提前结束
        if result['p3_accuracy'] > 0.65:
            print(f"\n🏆 超越目标！P3 accuracy: {result['p3_accuracy']:.6f} > 0.65")
            break

    # 总结
    print(f"\n{'='*60}")
    print("📊 实验总结")
    print(f"{'='*60}")

    for result in results:
        status = "🏆" if result['p3_accuracy'] > 0.63 else "✓" if result['p3_accuracy'] > 0.60 else "✗"
        print(f"{status} {result['name']}: {result['p3_accuracy']:.6f}")

    if best_result:
        print(f"\n🥇 最佳配置: {best_result['name']}")
        print(f"   P3 accuracy: {best_result['p3_accuracy']:.6f}")
        print(f"   结果文件: {best_result['result_file']}")

        # 保存最佳参数
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ultimate_p3_params_{timestamp}.py"

        with open(filename, "w") as f:
            f.write(f"# 高级P3优化最佳参数 - {timestamp}\n")
            f.write(f"# P3 accuracy: {best_result['p3_accuracy']:.6f}\n")
            f.write(f"# 配置名称: {best_result['name']}\n")
            f.write(f"# 结果文件: {best_result['result_file']}\n\n")
            f.write(f"BEST_P3_PARAMS = {best_result['params']}\n")

        print(f"\n✅ 最佳参数已保存到: {filename}")

    return best_result is not None

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)