#!/usr/bin/env python3
"""
验证 AVO_ULTRA 参数的可复现性
运行3次，评估稳定性
"""

import subprocess
import sys
import time
from datetime import datetime
import pandas as pd
import glob
import numpy as np

def get_latest_result():
    """获取最新结果文件"""
    files = sorted(glob.glob('tfdwt_detailed_results_*.csv'))
    if files:
        df = pd.read_csv(files[-1])
        return files[-1], df['p3_accuracy'].mean(), df['avo_accuracy'].mean(), df['overall_accuracy'].mean()
    return None, None, None, None

# AVO_ULTRA 最佳参数
BEST_PARAMS = {
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

def run_verification(run_num):
    """运行验证实验"""
    print(f"\n{'='*60}")
    print(f"🔬 验证运行 {run_num}/3")
    print(f"{'='*60}")

    params = BEST_PARAMS

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
        print("❌ 运行失败")
        return None

    result_file, p3_acc, avo_acc, overall_acc = get_latest_result()

    # 评估
    if avo_acc >= 0.63:
        marker = "🌟"
        status = "优秀"
    elif avo_acc >= 0.62:
        marker = "✅"
        status = "达标"
    elif avo_acc >= 0.60:
        marker = "⚠️"
        status = "接近"
    else:
        marker = "❌"
        status = "偏低"

    print(f"{marker} AVO: {avo_acc:.4f} | P3: {p3_acc:.4f} | Overall: {overall_acc:.4f} | {status}")
    print(f"⏱️  {duration/60:.1f}分钟")

    return {
        'run': run_num,
        'avo_accuracy': avo_acc,
        'p3_accuracy': p3_acc,
        'overall_accuracy': overall_acc,
        'result_file': result_file
    }

def main():
    print("🔍 验证 AVO_ULTRA 参数可复现性")
    print("="*60)
    print("参考结果: AVO准确率 = 0.6395")
    print("目标: AVO准确率 ≥ 0.62")
    print("验证次数: 3次")
    print("="*60)

    results = []

    for i in range(1, 4):
        result = run_verification(i)
        if result:
            results.append(result)
        else:
            print(f"⚠️  运行{i}失败")

    # 统计分析
    print(f"\n{'='*60}")
    print(f"📊 可复现性分析")
    print(f"{'='*60}")

    if not results:
        print("❌ 所有验证都失败")
        return False

    avo_accs = [r['avo_accuracy'] for r in results]
    p3_accs = [r['p3_accuracy'] for r in results]

    reference = 0.6395
    mean_avo = np.mean(avo_accs)
    std_avo = np.std(avo_accs)
    min_avo = np.min(avo_accs)
    max_avo = np.max(avo_accs)

    print(f"\nAVO准确率统计:")
    print(f"  参考值: {reference:.4f}")
    print(f"  平均值: {mean_avo:.4f}")
    print(f"  标准差: {std_avo:.4f}")
    print(f"  范围: {min_avo:.4f} - {max_avo:.4f}")
    print(f"  与参考值差异: {mean_avo - reference:+.4f}")
    print()

    # 达标情况
    success_count = sum(1 for x in avo_accs if x >= 0.62)
    success_rate = success_count / len(avo_accs)

    print(f"达标情况:")
    print(f"  ≥ 0.63 (优秀): {sum(1 for x in avo_accs if x >= 0.63)}/{len(avo_accs)}")
    print(f"  ≥ 0.62 (达标): {success_count}/{len(avo_accs)} ({success_rate*100:.1f}%)")
    print()

    # 可复现性判断
    if mean_avo >= 0.62 and success_rate >= 0.67:
        print("✅ 可复现性: 优秀 (平均达标且成功率≥67%)")
        reproducible = "优秀"
    elif mean_avo >= 0.61 and success_rate >= 0.5:
        print("✅ 可复现性: 良好 (平均接近且成功率≥50%)")
        reproducible = "良好"
    elif mean_avo >= 0.60:
        print("⚠️  可复现性: 一般 (平均≥0.60)")
        reproducible = "一般"
    else:
        print("❌ 可复现性: 较差")
        reproducible = "较差"

    # 保存验证报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"avo_verification_report_{timestamp}.txt"

    with open(report_file, "w") as f:
        f.write("P3_80_AVO_10 可复现性验证报告\n")
        f.write("="*60 + "\n\n")
        f.write(f"配置: AVO_ULTRA\n")
        f.write(f"参考结果: AVO准确率 = {reference:.4f}\n")
        f.write(f"验证时间: {timestamp}\n")
        f.write(f"验证次数: {len(results)}\n\n")

        f.write("参数配置:\n")
        for key, value in BEST_PARAMS.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        f.write("验证结果:\n")
        f.write(f"  AVO准确率平均: {mean_avo:.4f}\n")
        f.write(f"  AVO准确率标准差: {std_avo:.4f}\n")
        f.write(f"  AVO准确率范围: {min_avo:.4f} - {max_avo:.4f}\n")
        f.write(f"  达标率: {success_rate*100:.1f}%\n")
        f.write(f"  可复现性评级: {reproducible}\n\n")

        f.write("各次运行结果:\n")
        for r in results:
            f.write(f"  运行{r['run']}: AVO={r['avo_accuracy']:.4f}, P3={r['p3_accuracy']:.4f}\n")

    print(f"\n✅ 验证报告已保存: {report_file}")

    # 判断验证是否成功
    success = (mean_avo >= 0.61 and success_rate >= 0.5)

    if success:
        print("\n🎉 验证成功! 参数可复现")
    else:
        print("\n⚠️  验证显示较大变异性")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)