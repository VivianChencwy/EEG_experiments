#!/usr/bin/env python3
"""
精确复现0.6301成功结果的参数
"""

import subprocess
import sys
from datetime import datetime
import pandas as pd
import glob

def get_latest_result():
    files = sorted(glob.glob('tfdwt_detailed_results_*.csv'))
    if files:
        df = pd.read_csv(files[-1])
        return files[-1], df['p3_accuracy'].mean(), df['overall_accuracy'].mean()
    return None, None, None

# 精确的成功参数 (来自tfdwt_detailed_results_20250928_230851.csv)
SUCCESS_PARAMS = {
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

def main():
    print("🎯 复现0.6301成功结果")
    print("="*60)
    print("使用精确参数配置...")

    params = SUCCESS_PARAMS

    # 运行5次以获得稳定结果
    print("\n将运行5次实验以评估稳定性...")

    results = []

    for run in range(1, 6):
        print(f"\n{'='*60}")
        print(f"运行 {run}/5")
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

        cmd = f"python3 main_tfdwt.py combined P3_10 AVO_80 {param_str}"

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            result_file, p3_acc, overall_acc = get_latest_result()
            results.append({
                'run': run,
                'p3_accuracy': p3_acc,
                'overall_accuracy': overall_acc,
                'result_file': result_file
            })

            marker = "🎉" if p3_acc > 0.63 else "✅" if p3_acc > 0.60 else "⚠️"
            print(f"{marker} P3: {p3_acc:.6f} | Overall: {overall_acc:.6f}")
        else:
            print(f"❌ 运行失败")

    # 统计分析
    if results:
        print(f"\n{'='*60}")
        print("📊 稳定性分析")
        print(f"{'='*60}")

        p3_accs = [r['p3_accuracy'] for r in results]
        import numpy as np

        print(f"P3准确率统计:")
        print(f"  平均值: {np.mean(p3_accs):.6f}")
        print(f"  最大值: {np.max(p3_accs):.6f}")
        print(f"  最小值: {np.min(p3_accs):.6f}")
        print(f"  标准差: {np.std(p3_accs):.6f}")
        print(f"  超过0.60: {sum(1 for x in p3_accs if x > 0.60)}/{len(p3_accs)}")
        print(f"  超过0.63: {sum(1 for x in p3_accs if x > 0.63)}/{len(p3_accs)}")

        # 找到最佳结果
        best_idx = np.argmax(p3_accs)
        best = results[best_idx]

        print(f"\n🏆 最佳运行: 运行{best['run']}")
        print(f"   P3 accuracy: {best['p3_accuracy']:.6f}")
        print(f"   Overall accuracy: {best['overall_accuracy']:.6f}")
        print(f"   结果文件: {best['result_file']}")

        # 保存参数
        if best['p3_accuracy'] > 0.60:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ultimate_p3_params_{timestamp}.py"

            with open(filename, "w") as f:
                f.write(f"# 复现成功参数 - {timestamp}\n")
                f.write(f"# 最佳P3 accuracy: {best['p3_accuracy']:.6f}\n")
                f.write(f"# 平均P3 accuracy: {np.mean(p3_accs):.6f}\n")
                f.write(f"# 5次运行稳定性验证\n\n")
                f.write(f"VERIFIED_P3_PARAMS = {SUCCESS_PARAMS}\n\n")
                f.write(f"# 所有运行结果:\n")
                for r in results:
                    f.write(f"# Run {r['run']}: P3={r['p3_accuracy']:.6f}, Overall={r['overall_accuracy']:.6f}\n")

            print(f"\n✅ 参数已保存: {filename}")

if __name__ == "__main__":
    main()