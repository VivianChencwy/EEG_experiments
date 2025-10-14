#!/usr/bin/env python3
"""
并行调参功能 - 支持多GPU/多进程并行调参
"""

import os
import sys
import json
import time
import shutil
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

class ParallelTuner:
    def __init__(self, base_config_path: str = "config.py", n_workers: int = None):
        self.base_config_path = base_config_path
        self.n_workers = n_workers or min(4, mp.cpu_count())  # 默认使用4个进程或CPU核心数
        self.results_dir = Path("parallel_tuning_results")
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"并行调参初始化: {self.n_workers} 个工作进程")
    
    def create_worker_config(self, params: Dict[str, Any], worker_id: int, trial_id: int) -> str:
        """为工作进程创建配置文件"""
        # 读取基础配置
        with open(self.base_config_path, 'r') as f:
            content = f.read()
        
        # 修改关键参数
        modifications = {
            'MAX_EPOCHS = 500': f'MAX_EPOCHS = {params.get("MAX_EPOCHS", 100)}',
            'EARLY_STOPPING_PATIENCE = 50': f'EARLY_STOPPING_PATIENCE = {params.get("EARLY_STOPPING_PATIENCE", 20)}',
            'NESTED_CV_OUTER_FOLDS = 5': f'NESTED_CV_OUTER_FOLDS = {params.get("CV_FOLDS", 3)}',
            'NESTED_CV_REPEATS = 5': f'NESTED_CV_REPEATS = {params.get("CV_REPEATS", 2)}',
            'LEARNING_RATE = 0.01': f'LEARNING_RATE = {params.get("LEARNING_RATE", 0.01)}',
            'BATCH_SIZE = 32': f'BATCH_SIZE = {params.get("BATCH_SIZE", 32)}',
            'DROPOUT_RATE = 0.25': f'DROPOUT_RATE = {params.get("DROPOUT_RATE", 0.25)}',
            'WEIGHT_DECAY = 1e-4': f'WEIGHT_DECAY = {params.get("WEIGHT_DECAY", 1e-4)}',
            'classifier = \'EEGConformer\'': f'classifier = \'{params.get("classifier", "EEGConformer")}\'',
            'NOISE_STD = 0.005': f'NOISE_STD = {params.get("NOISE_STD", 0.005)}',
            'TIME_SHIFT_RANGE = 5': f'TIME_SHIFT_RANGE = {params.get("TIME_SHIFT_RANGE", 5)}',
            'LABEL_SMOOTHING = 0.05': f'LABEL_SMOOTHING = {params.get("LABEL_SMOOTHING", 0.05)}',
        }
        
        for old, new in modifications.items():
            content = content.replace(old, new)
        
        # 保存配置文件
        config_path = self.results_dir / f"config_worker_{worker_id}_trial_{trial_id}.py"
        with open(config_path, 'w') as f:
            f.write(content)
        
        return str(config_path)
    
    def run_worker_trial(self, args: Tuple[Dict[str, Any], int, int]) -> Dict[str, Any]:
        """工作进程运行单个试验"""
        params, worker_id, trial_id = args
        
        print(f"[Worker {worker_id}] 开始试验 {trial_id}")
        
        try:
            # 创建配置文件
            config_path = self.create_worker_config(params, worker_id, trial_id)
            
            # 设置环境变量
            env = os.environ.copy()
            env['CONFIG_OVERRIDE_PATH'] = config_path
            
            # 运行实验
            cmd = [sys.executable, "main_tfdwt.py"]
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=1800  # 30分钟超时
            )
            
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
                    'error': result.stderr[:200],
                    'params': params
                }
            
            # 提取准确率
            accuracy = self.extract_accuracy(result.stdout)
            
            print(f"[Worker {worker_id}] 试验 {trial_id} 完成 - 准确率: {accuracy:.4f}")
            
            return {
                'worker_id': worker_id,
                'trial_id': trial_id,
                'success': True,
                'accuracy': accuracy,
                'params': params,
                'stdout': result.stdout[:300]
            }

        except subprocess.TimeoutExpired:
            print(f"[Worker {worker_id}] 试验 {trial_id} 超时")
            return {
                'worker_id': worker_id,
                'trial_id': trial_id,
                'success': False,
                'accuracy': -1.0,
                'error': 'timeout',
                'params': params
            }
    except Exception as e:
        print(f"[Worker {worker_id}] 试验 {trial_id} 异常: {e}")
        return {
            'worker_id': worker_id,
            'trial_id': trial_id,
            'success': False,
            'accuracy': -1.0,
            'error': str(e),
            'params': params
        }
    
    def extract_accuracy(self, output: str) -> float:
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
    
    def generate_parameter_samples(self, n_trials: int) -> List[Dict[str, Any]]:
        """生成参数样本"""
        samples = []
        
        for _ in range(n_trials):
            sample = {
                'LEARNING_RATE': np.random.uniform(0.001, 0.05),
                'BATCH_SIZE': np.random.choice([16, 24, 32, 48, 64]),
                'DROPOUT_RATE': np.random.uniform(0.1, 0.4),
                'WEIGHT_DECAY': np.random.uniform(1e-5, 1e-2),
                'classifier': np.random.choice(['EEGConformer', 'EEGNetv4', 'SepConv1DLite', 'ShallowFBCSPNet']),
                'NOISE_STD': np.random.uniform(0.001, 0.02),
                'TIME_SHIFT_RANGE': np.random.randint(2, 15),
                'LABEL_SMOOTHING': np.random.uniform(0.0, 0.2),
                'MAX_EPOCHS': np.random.randint(50, 200),
                'EARLY_STOPPING_PATIENCE': np.random.randint(10, 30),
                'CV_FOLDS': np.random.choice([2, 3, 4]),
                'CV_REPEATS': np.random.choice([1, 2, 3]),
            }
            samples.append(sample)
        
        return samples
    
    def run_parallel_tuning(self, n_trials: int = 20):
        """运行并行调参"""
        print(f"🚀 开始并行调参: {n_trials} 个试验, {self.n_workers} 个工作进程")
        print("="*80)
        
        # 生成参数样本
        param_samples = self.generate_parameter_samples(n_trials)
        
        # 准备任务参数
        tasks = []
        for i, params in enumerate(param_samples):
            worker_id = i % self.n_workers
            tasks.append((params, worker_id, i))
        
        # 并行执行
        results = []
        best_accuracy = -1.0
        best_params = None
        completed_trials = 0
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # 提交所有任务
            future_to_task = {executor.submit(self.run_worker_trial, task): task for task in tasks}
            
            # 处理完成的任务
            for future in as_completed(future_to_task):
                result = future.result()
                results.append(result)
                completed_trials += 1
                
                # 更新最佳结果
                if result['success'] and result['accuracy'] > best_accuracy:
                    best_accuracy = result['accuracy']
                    best_params = result['params']
                    print(f"\n🎉 新最佳结果: {best_accuracy:.4f} (试验 {result['trial_id']})")
                
                # 实时进度报告
                elapsed_time = time.time() - start_time
                avg_time_per_trial = elapsed_time / completed_trials
                remaining_trials = n_trials - completed_trials
                estimated_remaining_time = remaining_trials * avg_time_per_trial
                
                print(f"进度: {completed_trials}/{n_trials} ({completed_trials/n_trials*100:.1f}%) | "
                      f"最佳: {best_accuracy:.4f} | "
                      f"预计剩余: {estimated_remaining_time/60:.1f}分钟")
        
        total_time = time.time() - start_time
        
        # 保存结果
        self._save_results(results, best_params, best_accuracy, total_time)
        
        print(f"\n🎉 并行调参完成！")
        print(f"总耗时: {total_time/60:.1f}分钟")
        print(f"最佳准确率: {best_accuracy:.4f}")
        print(f"成功试验: {sum(1 for r in results if r['success'])}/{len(results)}")
        
        return results, best_params, best_accuracy
    
    def _save_results(self, results: List[Dict], best_params: Dict, best_accuracy: float, total_time: float):
        """保存结果"""
        # 保存详细结果
        results_file = self.results_dir / "parallel_tuning_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'best_accuracy': best_accuracy,
                'best_params': best_params,
                'total_time': total_time,
                'n_workers': self.n_workers,
                'all_results': results,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        
        # 保存最佳参数配置
        if best_params:
            best_config_file = self.results_dir / "best_config.py"
            with open(self.base_config_path, 'r') as f:
                content = f.read()
            
            # 应用最佳参数
            modifications = {
                'LEARNING_RATE = 0.01': f'LEARNING_RATE = {best_params["LEARNING_RATE"]}',
                'BATCH_SIZE = 32': f'BATCH_SIZE = {best_params["BATCH_SIZE"]}',
                'DROPOUT_RATE = 0.25': f'DROPOUT_RATE = {best_params["DROPOUT_RATE"]}',
                'WEIGHT_DECAY = 1e-4': f'WEIGHT_DECAY = {best_params["WEIGHT_DECAY"]}',
                'classifier = \'EEGConformer\'': f'classifier = \'{best_params["classifier"]}\'',
                'NOISE_STD = 0.005': f'NOISE_STD = {best_params["NOISE_STD"]}',
                'TIME_SHIFT_RANGE = 5': f'TIME_SHIFT_RANGE = {best_params["TIME_SHIFT_RANGE"]}',
                'LABEL_SMOOTHING = 0.05': f'LABEL_SMOOTHING = {best_params["LABEL_SMOOTHING"]}',
            }
            
            for old, new in modifications.items():
                content = content.replace(old, new)
            
            with open(best_config_file, 'w') as f:
                f.write(content)
        
        print(f"结果已保存到: {self.results_dir}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='并行调参')
    parser.add_argument('--trials', type=int, default=20, help='试验数量')
    parser.add_argument('--workers', type=int, default=None, help='工作进程数')

    args = parser.parse_args()

    tuner = ParallelTuner(n_workers=args.workers)
    tuner.run_parallel_tuning(args.trials)

if __name__ == "__main__":
    main()