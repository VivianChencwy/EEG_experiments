#!/usr/bin/env python3
"""
快速并行调参测试 - 验证您的设备上的并行调参效率
"""

import os
import sys
import time
import subprocess
import multiprocessing as mp
from pathlib import Path

def run_single_trial(params, trial_id, worker_id):
    """运行单个试验"""
    print(f"[Worker {worker_id}] 开始试验 {trial_id}")
    
    try:
        # 创建简化的配置文件
        config_content = f"""
# 快速测试配置
P3_DATA_DIR = '../P3_Raw_Data_BIDS-Compatible'
AVO_DATA_DIR = '../ds005863'
use_combined_datasets = True
NESTED_CV_TRIALS_PER_SUBJECT_P3 = 10
NESTED_CV_TRIALS_PER_SUBJECT_AVO = 50
TRAIN_SIZE = 0.7
VAL_SIZE = 0.1
BATCH_SIZE = {params.get('BATCH_SIZE', 32)}
MAX_EPOCHS = {params.get('MAX_EPOCHS', 10)}
LEARNING_RATE = {params.get('LEARNING_RATE', 0.01)}
WEIGHT_DECAY = {params.get('WEIGHT_DECAY', 1e-4)}
DROPOUT_RATE = {params.get('DROPOUT_RATE', 0.25)}
EARLY_STOPPING_PATIENCE = 5
NOISE_STD = {params.get('NOISE_STD', 0.005)}
TIME_SHIFT_RANGE = {params.get('TIME_SHIFT_RANGE', 5)}
LABEL_SMOOTHING = {params.get('LABEL_SMOOTHING', 0.05)}
classifier = '{params.get('classifier', 'EEGConformer')}'
ELECTRODE_FUSION_METHOD = 'none'
DOMAIN_ADAPTATION_METHOD = 'none'
USE_ENHANCED_PREPROCESSING = True
seeds = [42]
NESTED_CV_OUTER_FOLDS = 2
NESTED_CV_REPEATS = 1
NESTED_CV_CONFIDENCE_LEVEL = 0.95
DEVICE_MODE = 'cuda'
"""
        
        config_path = f"temp_config_worker_{worker_id}_trial_{trial_id}.py"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        # 设置环境变量
        env = os.environ.copy()
        env['CONFIG_OVERRIDE_PATH'] = config_path
        
        # 运行实验
        start_time = time.time()
        cmd = [sys.executable, "main_tfdwt.py"]
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        duration = time.time() - start_time
        
        # 清理配置文件
        if os.path.exists(config_path):
            os.unlink(config_path)
        
        if result.returncode != 0:
            print(f"[Worker {worker_id}] 试验 {trial_id} 失败")
            return {
                'worker_id': worker_id,
                'trial_id': trial_id,
                'success': False,
                'accuracy': -1.0,
                'duration': duration,
                'error': result.stderr[:200]
            }
        
        # 提取准确率
        accuracy = extract_accuracy(result.stdout)
        
        print(f"[Worker {worker_id}] 试验 {trial_id} 完成 - 准确率: {accuracy:.4f}, 耗时: {duration:.1f}秒")
        
        return {
            'worker_id': worker_id,
            'trial_id': trial_id,
            'success': True,
            'accuracy': accuracy,
            'duration': duration,
            'params': params
        }
        
    except subprocess.TimeoutExpired:
        print(f"[Worker {worker_id}] 试验 {trial_id} 超时")
        return {
            'worker_id': worker_id,
            'trial_id': trial_id,
            'success': False,
            'accuracy': -1.0,
            'duration': 300,
            'error': 'timeout'
        }
    except Exception as e:
        print(f"[Worker {worker_id}] 试验 {trial_id} 异常: {e}")
        return {
            'worker_id': worker_id,
            'trial_id': trial_id,
            'success': False,
            'accuracy': -1.0,
            'duration': 0,
            'error': str(e)
        }

def extract_accuracy(output):
    """从输出中提取准确率"""
    import re
    
    patterns = [
        r'Overall accuracy:\s+([0-9.]+)',
        r'mean_accuracy[\'\"]*\s*[:\s=]+\s*([0-9.]+)',
        r'Final Results: Overall Accuracy = ([0-9.]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            accuracies = [float(match) for match in matches if float(match) <= 1.0]
            if accuracies:
                return max(accuracies)
    
    return 0.0

def run_parallel_test(n_trials=3, n_workers=4):
    """运行并行测试"""
    print(f"🚀 开始并行调参测试")
    print(f"试验数: {n_trials}, 工作进程数: {n_workers}")
    print("="*60)
    
    # 生成测试参数
    test_params = []
    for i in range(n_trials):
        params = {
            'LEARNING_RATE': 0.01,
            'BATCH_SIZE': 32,
            'DROPOUT_RATE': 0.25,
            'WEIGHT_DECAY': 1e-4,
            'classifier': 'EEGConformer',
            'NOISE_STD': 0.005,
            'TIME_SHIFT_RANGE': 5,
            'LABEL_SMOOTHING': 0.05,
            'MAX_EPOCHS': 10
        }
        test_params.append((params, i, i % n_workers))
    
    # 并行执行
    start_time = time.time()
    
    with mp.Pool(processes=n_workers) as pool:
        results = pool.starmap(run_single_trial, test_params)
    
    total_time = time.time() - start_time
    
    # 分析结果
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"\n📊 测试结果分析")
    print("="*60)
    print(f"总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"成功试验: {len(successful_results)}/{len(results)}")
    print(f"失败试验: {len(failed_results)}")
    
    if successful_results:
        accuracies = [r['accuracy'] for r in successful_results]
        durations = [r['duration'] for r in successful_results]
        
        print(f"平均准确率: {sum(accuracies)/len(accuracies):.4f}")
        print(f"最佳准确率: {max(accuracies):.4f}")
        print(f"平均每试验耗时: {sum(durations)/len(durations):.1f}秒")
        
        # 计算并行效率
        single_trial_time = sum(durations) / len(durations)
        parallel_efficiency = n_workers * single_trial_time / total_time
        speedup = n_workers * parallel_efficiency
        
        print(f"并行效率: {parallel_efficiency:.2f}")
        print(f"加速倍数: {speedup:.1f}x")
        
        # 预测完整调参时间
        full_trials = 20
        estimated_full_time = total_time * full_trials / n_trials
        
        print(f"\n📈 完整调参预测 (20个试验):")
        print(f"预计时间: {estimated_full_time:.1f}秒 ({estimated_full_time/60:.1f}分钟)")
        
        if estimated_full_time < 7200:  # 2小时
            print("✅ 您的设备可以高效运行并行调参！")
            return True, total_time, parallel_efficiency
        elif estimated_full_time < 14400:  # 4小时
            print("⚠️ 您的设备可以运行并行调参，但效率一般")
            return True, total_time, parallel_efficiency
        else:
            print("❌ 您的设备运行并行调参效率较低")
            return False, total_time, parallel_efficiency
    else:
        print("❌ 所有试验都失败了")
        return False, total_time, 0

def main():
    """主函数"""
    print("🔍 设备并行调参效率测试")
    print("="*60)
    
    # 检查系统配置
    cpu_cores = mp.cpu_count()
    print(f"CPU核心数: {cpu_cores}")
    
    # 推荐工作进程数
    recommended_workers = min(cpu_cores, 8)
    print(f"推荐工作进程数: {recommended_workers}")
    
    # 运行测试
    success, duration, efficiency = run_parallel_test(n_trials=3, n_workers=4)
    
    if success:
        print(f"\n🎉 测试成功！")
        print(f"实际效率: {efficiency:.2f}")
        print(f"建议使用并行调参策略")
    else:
        print(f"\n❌ 测试失败")
        print(f"建议使用分层调参或增量调参策略")

if __name__ == "__main__":
    main()
