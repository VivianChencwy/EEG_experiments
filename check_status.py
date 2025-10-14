#!/usr/bin/env python3
"""
检查调参系统状态和结果
"""

import os
import json
import glob
from pathlib import Path
from datetime import datetime

def check_tuning_status():
    """检查当前调参状态"""
    print("🔍 TF-DWT调参系统状态检查")
    print("=" * 60)

    # 检查结果目录
    result_dirs = [
        "quick_test_results",
        "standard_tuning_results",
        "extensive_tuning_results",
        "parallel_tuning_results",
        "focused_tuning_results",
        "quick_progress_results"
    ]

    found_results = False

    for result_dir in result_dirs:
        if Path(result_dir).exists():
            print(f"\n📁 发现结果目录: {result_dir}")

            # 检查结果文件
            results_file = Path(result_dir) / "tuning_results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        data = json.load(f)

                    n_trials = data.get('n_trials', 0)
                    best_score = data.get('best_score', -1.0)

                    print(f"   📊 试验完成数: {n_trials}")
                    if best_score > 0:
                        print(f"   🎯 最佳准确率: {best_score:.4f}")
                    else:
                        print(f"   ⚠️  最佳准确率: 未找到有效结果")

                    # 检查最佳配置
                    best_config = Path(result_dir) / "best_config.py"
                    if best_config.exists():
                        print(f"   ✅ 最佳配置已保存: {best_config}")

                    found_results = True

                except Exception as e:
                    print(f"   ❌ 读取结果失败: {e}")
            else:
                print(f"   📝 结果文件不存在")

    # 检查运行中的进程
    print(f"\n🏃 检查运行中的调参进程...")
    import subprocess
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        tuning_processes = [line for line in result.stdout.split('\n')
                          if 'tune_tfdwt' in line or 'main_tfdwt' in line]

        if tuning_processes:
            print(f"   🔄 发现 {len(tuning_processes)} 个运行中的进程")
            for proc in tuning_processes:
                print(f"      {proc.strip()}")
        else:
            print(f"   💤 当前没有运行中的调参进程")
    except:
        print(f"   ⚠️  无法检查进程状态")

    # 检查最新日志
    print(f"\n📝 检查最新训练日志...")
    log_files = []
    for log_dir in ["log_0909", "log_batch"]:
        if Path(log_dir).exists():
            log_files.extend(list(Path(log_dir).glob("TF_DWT*.log")))

    if log_files:
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        mod_time = datetime.fromtimestamp(latest_log.stat().st_mtime)
        print(f"   📄 最新日志: {latest_log}")
        print(f"   ⏰ 修改时间: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 检查日志中的最新准确率
        try:
            with open(latest_log, 'r') as f:
                content = f.read()

            import re
            matches = re.findall(r'Overall accuracy:\s+([0-9.]+)', content)
            if matches:
                latest_acc = float(matches[-1])
                print(f"   🎯 最新准确率: {latest_acc:.4f}")
        except:
            pass
    else:
        print(f"   📭 未找到训练日志")

    # 总结和建议
    print(f"\n💡 建议:")
    if not found_results:
        print(f"   🚀 还没有调参结果，建议运行:")
        print(f"      python run_ultrafast_test.py  (验证系统)")
        print(f"      python run_tuning_example.py --mode quick  (快速调参)")
    else:
        print(f"   ✅ 已有调参结果")
        print(f"   📋 查看详细结果: cat */tuning_results.json")
        print(f"   🔧 应用最佳参数: cp */best_config.py config.py")

    print(f"\n📚 完整教程: cat TUNING_TUTORIAL.md")
    print("=" * 60)

if __name__ == "__main__":
    check_tuning_status()