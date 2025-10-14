#!/usr/bin/env python3
"""
验证0.6301结果的可复现性
使用相同参数运行多次，评估结果的稳定性和可复现性
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
        return files[-1], df['p3_accuracy'].mean(), df['overall_accuracy'].mean(), df
    return None, None, None, None

# 成功的参数配置 (来自230851)
PROVEN_PARAMS = {
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

def run_single_experiment(run_num):
    """运行单次实验"""
    print(f"\n{'='*60}")
    print(f"🔬 运行 {run_num}")
    print(f"{'='*60}")

    params = PROVEN_PARAMS

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
    duration = time.time() - start_time

    if result.returncode != 0:
        print(f"❌ 运行失败")
        return None

    result_file, p3_acc, overall_acc, df = get_latest_result()

    # 评估结果质量
    if p3_acc >= 0.625:
        marker = "🌟"
        status = "优秀"
    elif p3_acc >= 0.60:
        marker = "✅"
        status = "良好"
    elif p3_acc >= 0.58:
        marker = "⚠️"
        status = "一般"
    else:
        marker = "❌"
        status = "偏低"

    print(f"{marker} P3: {p3_acc:.6f} | Overall: {overall_acc:.6f} | {duration/60:.1f}分钟 | {status}")

    return {
        'run': run_num,
        'p3_accuracy': p3_acc,
        'overall_accuracy': overall_acc,
        'result_file': result_file,
        'duration': duration,
        'df': df
    }

def main():
    print("🔍 验证0.6301结果的可复现性")
    print("="*60)
    print("目标: 验证P3准确率能稳定在0.60左右")
    print("参考结果: 0.6301 (230851)")
    print("运行次数: 10次")
    print("="*60)

    NUM_RUNS = 10
    results = []

    for i in range(1, NUM_RUNS + 1):
        result = run_single_experiment(i)
        if result:
            results.append(result)
        else:
            print(f"⚠️  运行{i}失败，跳过")

        # 每3次显示一次中期统计
        if i % 3 == 0 and results:
            p3_accs = [r['p3_accuracy'] for r in results]
            print(f"\n   中期统计 ({len(results)}次): 平均P3={np.mean(p3_accs):.4f}, 范围={np.min(p3_accs):.4f}-{np.max(p3_accs):.4f}")

    # 最终统计分析
    print(f"\n{'='*60}")
    print(f"📊 可复现性分析 (完成{len(results)}次运行)")
    print(f"{'='*60}")

    if not results:
        print("❌ 所有运行都失败")
        return False

    p3_accs = [r['p3_accuracy'] for r in results]
    overall_accs = [r['overall_accuracy'] for r in results]

    print(f"\nP3准确率统计:")
    print(f"  参考值 (230851): 0.6301")
    print(f"  平均值: {np.mean(p3_accs):.6f}")
    print(f"  中位数: {np.median(p3_accs):.6f}")
    print(f"  标准差: {np.std(p3_accs):.6f}")
    print(f"  最小值: {np.min(p3_accs):.6f}")
    print(f"  最大值: {np.max(p3_accs):.6f}")
    print(f"  范围: {np.max(p3_accs) - np.min(p3_accs):.6f}")
    print()

    print(f"Overall准确率统计:")
    print(f"  参考值 (230851): 0.6378")
    print(f"  平均值: {np.mean(overall_accs):.6f}")
    print(f"  标准差: {np.std(overall_accs):.6f}")
    print()

    # 评估可复现性
    print("可复现性评估:")
    print(f"  ≥ 0.625 (优秀): {sum(1 for x in p3_accs if x >= 0.625)}/{len(p3_accs)} ({100*sum(1 for x in p3_accs if x >= 0.625)/len(p3_accs):.1f}%)")
    print(f"  ≥ 0.60 (良好): {sum(1 for x in p3_accs if x >= 0.60)}/{len(p3_accs)} ({100*sum(1 for x in p3_accs if x >= 0.60)/len(p3_accs):.1f}%)")
    print(f"  ≥ 0.58 (一般): {sum(1 for x in p3_accs if x >= 0.58)}/{len(p3_accs)} ({100*sum(1 for x in p3_accs if x >= 0.58)/len(p3_accs):.1f}%)")
    print()

    # 与参考值比较
    reference = 0.6301
    mean_p3 = np.mean(p3_accs)
    diff_from_ref = mean_p3 - reference

    print(f"与参考值比较:")
    print(f"  参考值: {reference:.6f}")
    print(f"  平均值: {mean_p3:.6f}")
    print(f"  差异: {diff_from_ref:+.6f} ({diff_from_ref/reference*100:+.2f}%)")
    print()

    # 可复现性判断
    if abs(diff_from_ref) <= 0.02:
        print("✅ 可复现性: 优秀 (差异≤2%)")
        reproducible = "优秀"
    elif abs(diff_from_ref) <= 0.03:
        print("✅ 可复现性: 良好 (差异≤3%)")
        reproducible = "良好"
    elif abs(diff_from_ref) <= 0.05:
        print("⚠️  可复现性: 一般 (差异≤5%)")
        reproducible = "一般"
    else:
        print("❌ 可复现性: 较差 (差异>5%)")
        reproducible = "较差"

    # 找到最佳运行
    best_idx = np.argmax(p3_accs)
    best = results[best_idx]

    print(f"\n🏆 最佳运行: 运行{best['run']}")
    print(f"   P3 accuracy: {best['p3_accuracy']:.6f}")
    print(f"   Overall accuracy: {best['overall_accuracy']:.6f}")
    print(f"   结果文件: {best['result_file']}")

    # 保存验证报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"reproducibility_report_{timestamp}.txt"

    with open(report_file, "w") as f:
        f.write("可复现性验证报告\n")
        f.write("="*60 + "\n\n")
        f.write(f"参考结果: P3=0.6301 (tfdwt_detailed_results_20250928_230851.csv)\n")
        f.write(f"验证时间: {timestamp}\n")
        f.write(f"运行次数: {len(results)}\n\n")

        f.write("参数配置:\n")
        for key, value in PROVEN_PARAMS.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        f.write("统计结果:\n")
        f.write(f"  P3准确率平均: {np.mean(p3_accs):.6f}\n")
        f.write(f"  P3准确率中位数: {np.median(p3_accs):.6f}\n")
        f.write(f"  P3准确率标准差: {np.std(p3_accs):.6f}\n")
        f.write(f"  P3准确率范围: {np.min(p3_accs):.6f} - {np.max(p3_accs):.6f}\n")
        f.write(f"  与参考值差异: {diff_from_ref:+.6f} ({diff_from_ref/reference*100:+.2f}%)\n")
        f.write(f"  可复现性评级: {reproducible}\n\n")

        f.write("所有运行结果:\n")
        for r in sorted(results, key=lambda x: x['p3_accuracy'], reverse=True):
            f.write(f"  运行{r['run']}: P3={r['p3_accuracy']:.6f}, Overall={r['overall_accuracy']:.6f}\n")

    print(f"\n✅ 验证报告已保存: {report_file}")

    # 判断是否成功验证
    success = (mean_p3 >= 0.58 and abs(diff_from_ref) <= 0.05)

    if success:
        print("\n🎉 验证成功! 结果基本可复现")
    else:
        print("\n⚠️  验证结果显示较大变异性")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)