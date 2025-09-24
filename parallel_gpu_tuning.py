#!/usr/bin/env python3
"""
GPU优化的并行调参 - 专为RTX 4090等高端GPU设计

这个脚本会：
1. 检测GPU内存容量
2. 根据GPU内存智能调整并行数
3. 确保每个进程都能充分利用GPU
"""

import argparse
import subprocess
import sys
import time
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List
import numpy as np

def get_gpu_info():
    """获取GPU信息"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                name, memory = line.split(', ')
                gpu_info.append({'name': name.strip(), 'memory_mb': int(memory)})
            return gpu_info
    except:
        pass
    return []

def calculate_optimal_parallel_count(gpu_memory_mb: int) -> int:
    """根据GPU内存计算最优并行数"""
    # RTX 4090 (24GB) 可以同时运行2-3个EEG模型
    # 每个EEG模型大约需要6-8GB显存
    if gpu_memory_mb >= 20000:  # >= 20GB (RTX 4090类)
        return 3
    elif gpu_memory_mb >= 16000:  # >= 16GB (RTX 4080类)
        return 2
    elif gpu_memory_mb >= 12000:  # >= 12GB (RTX 4070Ti类)
        return 2
    elif gpu_memory_mb >= 8000:   # >= 8GB (RTX 4070类)
        return 1
    else:
        return 1

def run_tuning_subprocess_gpu(process_id: int, n_trials: int, base_seed: int, gpu_id: int = 0) -> Dict[str, Any]:
    """在指定GPU上运行调参子进程"""

    # 设置GPU环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # 为每个进程创建独立的结果目录
    results_dir = f"gpu_parallel_results/process_{process_id}"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # 设置随机种子
    np.random.seed(base_seed + process_id * 1000)

    print(f"🚀 启动GPU进程 {process_id} (GPU {gpu_id}) - {n_trials} trials")

    # 运行调参
    cmd = [
        sys.executable, "tune_tfdwt.py",
        "--strategy", "random",
        "--n_trials", str(n_trials),
        "--results_dir", results_dir
    ]

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=14400  # 4小时超时
        )
        duration = time.time() - start_time

        if result.returncode == 0:
            # 加载结果
            results_file = Path(results_dir) / "tuning_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    tuning_results = json.load(f)

                return {
                    'process_id': process_id,
                    'gpu_id': gpu_id,
                    'success': True,
                    'duration': duration,
                    'best_score': tuning_results.get('best_score', 0.0),
                    'best_params': tuning_results.get('best_params', {}),
                    'n_trials': tuning_results.get('n_trials', 0),
                    'results_file': str(results_file)
                }
            else:
                return {
                    'process_id': process_id,
                    'gpu_id': gpu_id,
                    'success': False,
                    'duration': duration,
                    'error': 'Results file not found'
                }
        else:
            return {
                'process_id': process_id,
                'gpu_id': gpu_id,
                'success': False,
                'duration': duration,
                'error': result.stderr
            }

    except subprocess.TimeoutExpired:
        return {
            'process_id': process_id,
            'gpu_id': gpu_id,
            'success': False,
            'duration': time.time() - start_time,
            'error': 'Process timeout'
        }
    except Exception as e:
        return {
            'process_id': process_id,
            'gpu_id': gpu_id,
            'success': False,
            'duration': time.time() - start_time,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='GPU优化的并行超参数调参')
    parser.add_argument('--n_processes', type=int, default=None,
                        help='并行进程数 (默认根据GPU自动检测)')
    parser.add_argument('--trials_per_process', type=int, default=25,
                        help='每个进程的试验数')
    parser.add_argument('--base_seed', type=int, default=42,
                        help='随机种子基数')

    args = parser.parse_args()

    # 检测GPU
    gpu_info = get_gpu_info()
    if not gpu_info:
        print("❌ 未检测到NVIDIA GPU，请使用CPU版本")
        return 1

    print("🔍 GPU信息检测:")
    for i, gpu in enumerate(gpu_info):
        print(f"   GPU {i}: {gpu['name']} ({gpu['memory_mb']/1024:.1f}GB)")

    primary_gpu = gpu_info[0]

    # 计算最优并行数
    if args.n_processes is None:
        optimal_processes = calculate_optimal_parallel_count(primary_gpu['memory_mb'])
        print(f"🧠 根据GPU内存自动设置并行数: {optimal_processes}")
    else:
        optimal_processes = args.n_processes
        print(f"🔧 手动设置并行数: {optimal_processes}")

    total_trials = optimal_processes * args.trials_per_process

    print(f"\n🎯 GPU并行调参配置:")
    print(f"   GPU: {primary_gpu['name']} ({primary_gpu['memory_mb']/1024:.1f}GB)")
    print(f"   并行进程数: {optimal_processes}")
    print(f"   每进程试验数: {args.trials_per_process}")
    print(f"   总试验数: {total_trials}")
    print(f"   预计时间: {total_trials * 20 / 60 / optimal_processes:.1f}-{total_trials * 30 / 60 / optimal_processes:.1f} 小时")
    print(f"   结果目录: gpu_parallel_results/")
    print("="*60)

    # 确认运行
    response = input("🚀 确认开始GPU并行调参吗? (y/N): ")
    if response.lower() != 'y':
        print("❌ 取消运行")
        return 0

    start_time = time.time()

    # 并行运行
    with ProcessPoolExecutor(max_workers=optimal_processes) as executor:
        # 提交所有进程，都使用主GPU
        future_to_process = {
            executor.submit(run_tuning_subprocess_gpu, i, args.trials_per_process, args.base_seed, 0): i
            for i in range(optimal_processes)
        }

        process_results = []
        completed = 0

        for future in as_completed(future_to_process):
            process_id = future_to_process[future]
            try:
                result = future.result()
                process_results.append(result)
                completed += 1

                status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
                elapsed = time.time() - start_time
                print(f"\n🔄 进程 {process_id} 完成 ({completed}/{optimal_processes}) - {status}")
                print(f"   GPU: {result.get('gpu_id', 0)} | 耗时: {elapsed/60:.1f}分钟")

                if result['success']:
                    print(f"   最佳准确率: {result['best_score']:.4f}")
                else:
                    print(f"   错误: {result.get('error', 'Unknown')}")

            except Exception as e:
                print(f"❌ 进程 {process_id} 异常: {e}")
                process_results.append({
                    'process_id': process_id,
                    'gpu_id': 0,
                    'success': False,
                    'error': str(e),
                    'duration': time.time() - start_time
                })

    # 合并结果
    from parallel_tuning import combine_results, save_combined_results
    combined_results = combine_results(process_results)
    results_file, report_file = save_combined_results(combined_results, "gpu_parallel_results")

    total_time = time.time() - start_time

    # 最终报告
    print(f"\n{'='*60}")
    print("🎉 GPU并行调参完成!")
    print(f"{'='*60}")
    print(f"⏰ 总耗时: {total_time/60:.1f} 分钟 ({total_time/3600:.2f} 小时)")
    print(f"🚀 成功进程: {combined_results['successful_processes']}/{combined_results['total_processes']}")
    print(f"📊 总试验数: {len([t for t in combined_results['all_trials'] if t['score'] > 0])}")
    print(f"🏆 最佳准确率: {combined_results['best_score']:.4f}")
    print(f"📁 结果保存到: {results_file}")
    print(f"📝 报告保存到: {report_file}")

    if combined_results['best_params']:
        print(f"🔧 最佳配置: gpu_parallel_results/best_config_parallel.py")
        print(f"\n💡 应用最佳参数:")
        print(f"   cp gpu_parallel_results/best_config_parallel.py config.py")

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)