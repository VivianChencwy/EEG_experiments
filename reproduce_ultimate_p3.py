#!/usr/bin/env python3
"""
复现终极P3优化结果的脚本
P3准确率目标：> 0.6
预期结果：P3准确率 = 0.6301

使用方法：
python3 reproduce_ultimate_p3.py
"""

import subprocess
import sys
from optimal_tfdwt_parameters_backup import ULTIMATE_SUCCESS_P3_PARAMS

def main():
    print("🚀 开始复现终极P3优化结果")
    print("="*50)
    print("数据集配置：P3=10, AVO=80")
    print("目标：P3准确率 > 0.6")
    print("预期结果：P3准确率 = 0.6301")
    print()

    # 从备份文件获取成功参数
    params = ULTIMATE_SUCCESS_P3_PARAMS
    print("使用的参数配置：")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()

    # 构建命令行参数
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

    # 运行实验
    cmd = f"python3 main_tfdwt.py combined P3_10 AVO_80 {param_str}"
    print("执行命令：")
    print(cmd)
    print()
    print("⏱️  开始执行（预计需要30-60分钟）...")

    result = subprocess.run(cmd, shell=True)

    if result.returncode == 0:
        print("✅ 实验执行完成！")
        print("请检查最新的 tfdwt_detailed_results_*.csv 文件查看结果")

        # 自动分析结果
        print("\n📊 自动分析最新结果...")
        analysis_cmd = """
python3 -c "
import pandas as pd
import glob

files = sorted(glob.glob('tfdwt_detailed_results_*.csv'))
if files:
    latest_file = files[-1]
    df = pd.read_csv(latest_file)
    p3_acc = df['p3_accuracy'].mean()
    print(f'最新结果文件: {latest_file}')
    print(f'P3准确率: {p3_acc:.4f}')
    print(f'目标达成: {\"✅ 是\" if p3_acc > 0.6 else \"❌ 否\"}')
    print(f'预期结果复现: {\"✅ 成功\" if abs(p3_acc - 0.6301) < 0.02 else \"❌ 有差异\"}')
else:
    print('未找到结果文件')
"
        """
        subprocess.run(analysis_cmd, shell=True)

    else:
        print("❌ 实验执行失败，请检查错误信息")
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)