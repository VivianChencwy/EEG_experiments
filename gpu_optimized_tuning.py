#!/usr/bin/env python3
"""
GPU优化调参 - 最大化GPU利用率的调参策略
"""

import os
import sys
import json
import time
import shutil
import subprocess
import threading
import queue
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
from tqdm import tqdm

class GPUOptimizedTuner:
    def __init__(self, base_config_path: str = "config.py"):
        self.base_config_path = base_config_path
        self.results_dir = Path("gpu_optimized_tuning_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # GPU配置
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        self.gpu_memory_gb = self._get_gpu_memory()
        
        print(f"GPU优化调参初始化")
        print(f"GPU可用: {self.gpu_available}")
        if self.gpu_available:
            print(f"GPU数量: {self.gpu_count}")
            print(f"GPU内存: {self.gpu_memory_gb:.1f}GB")
            for i in range(self.gpu_count):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    def _get_gpu_memory(self):
        """获取GPU内存大小"""
        if not self.gpu_available:
            return 0
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    def create_gpu_optimized_config(self, params: Dict[str, Any], trial_id: int) -> str:
        """创建GPU优化的配置文件"""
        # 读取基础配置
        with open(self.base_config_path, 'r') as f:
            content = f.read()
        
        # GPU优化配置
        gpu_optimizations = {
            'DEVICE_MODE = \'auto\'': 'DEVICE_MODE = \'cuda\'',  # 强制使用GPU
            'BATCH_SIZE = 32': f'BATCH_SIZE = {params.get("BATCH_SIZE", 64)}',  # 增大batch size
            'MAX_EPOCHS = 500': f'MAX_EPOCHS = {params.get("MAX_EPOCHS", 200)}',
            'EARLY_STOPPING_PATIENCE = 50': f'EARLY_STOPPING_PATIENCE = {params.get("EARLY_STOPPING_PATIENCE", 30)}',
            'NESTED_CV_OUTER_FOLDS = 5': f'NESTED_CV_OUTER_FOLDS = {params.get("CV_FOLDS", 3)}',
            'NESTED_CV_REPEATS = 5': f'NESTED_CV_REPEATS = {params.get("CV_REPEATS", 2)}',
            'LEARNING_RATE = 0.01': f'LEARNING_RATE = {params.get("LEARNING_RATE", 0.01)}',
            'DROPOUT_RATE = 0.25': f'DROPOUT_RATE = {params.get("DROPOUT_RATE", 0.25)}',
            'WEIGHT_DECAY = 1e-4': f'WEIGHT_DECAY = {params.get("WEIGHT_DECAY", 1e-4)}',
            'classifier = \'EEGConformer\'': f'classifier = \'{params.get("classifier", "EEGConformer")}\'',
            'NOISE_STD = 0.005': f'NOISE_STD = {params.get("NOISE_STD", 0.005)}',
            'TIME_SHIFT_RANGE = 5': f'TIME_SHIFT_RANGE = {params.get("TIME_SHIFT_RANGE", 5)}',
            'LABEL_SMOOTHING = 0.05': f'LABEL_SMOOTHING = {params.get("LABEL_SMOOTHING", 0.05)}',
        }
        
        # 添加GPU优化参数
        gpu_optimizations.update({
            '# GPU优化参数': '# GPU优化参数',
            'USE_MIXED_PRECISION = False': 'USE_MIXED_PRECISION = True',  # 混合精度训练
            'GPU_MEMORY_FRACTION = 0.9': 'GPU_MEMORY_FRACTION = 0.95',  # 使用更多GPU内存
            'ENABLE_GPU_OPTIMIZATION = False': 'ENABLE_GPU_OPTIMIZATION = True',
        })
        
        for old, new in gpu_optimizations.items():
            content = content.replace(old, new)
        
        # 保存配置文件
        config_path = self.results_dir / f"gpu_config_trial_{trial_id}.py"
        with open(config_path, 'w') as f:
            f.write(content)
        
        return str(config_path)
    
    def run_gpu_trial(self, params: Dict[str, Any], trial_id: int, pbar: tqdm) -> Dict[str, Any]:
        """运行GPU优化的单个试验"""
        pbar.set_description(f"试验 {trial_id + 1}")
        pbar.set_postfix_str("准备中...")
        
        start_time = time.time()
        
        try:
            # 创建GPU优化配置
            pbar.set_postfix_str("创建配置...")
            config_path = self.create_gpu_optimized_config(params, trial_id)
            
            # 设置环境变量
            env = os.environ.copy()
            env['CONFIG_OVERRIDE_PATH'] = config_path
            env['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一个GPU
            
            # 运行实验
            pbar.set_postfix_str("运行训练...")
            cmd = [sys.executable, "main_tfdwt.py"]
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=1800  # 30分钟超时
            )
            
            duration = time.time() - start_time
            
            # 清理配置文件
            if os.path.exists(config_path):
                os.unlink(config_path)
            
            if result.returncode != 0:
                pbar.set_postfix_str(f"失败: {result.stderr[:50]}...")
                return {
                    'trial_id': trial_id,
                    'success': False,
                    'accuracy': -1.0,
                    'error': result.stderr,
                    'duration': duration
                }
            
            # 提取准确率
            accuracy = self.extract_accuracy(result.stdout)
            
            pbar.set_postfix_str(f"准确率: {accuracy:.4f}, 耗时: {duration/60:.1f}分钟")
            
            return {
                'trial_id': trial_id,
                'success': True,
                'accuracy': accuracy,
                'duration': duration,
                'params': params,
                'stdout': result.stdout[:500]
            }
            
        except subprocess.TimeoutExpired:
            pbar.set_postfix_str("超时")
            return {'trial_id': trial_id, 'success': False, 'accuracy': -1.0, 'error': 'timeout'}
        except Exception as e:
            pbar.set_postfix_str(f"异常: {str(e)[:30]}...")
            return {'trial_id': trial_id, 'success': False, 'accuracy': -1.0, 'error': str(e)}
    
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
    
    def generate_layered_params(self, stage: str, n_trials: int, best_params: Dict = None) -> List[Dict[str, Any]]:
        """生成分层调参的参数样本"""
        samples = []
        
        for _ in range(n_trials):
            if stage == "coarse":
                # 第一阶段：重要参数粗调
                sample = {
                    'LEARNING_RATE': np.random.uniform(0.001, 0.1),  # 大范围搜索
                    'BATCH_SIZE': np.random.choice([32, 64, 128, 256]),  # 基础batch size
                    'DROPOUT_RATE': np.random.uniform(0.1, 0.6),  # 大范围
                    'classifier': np.random.choice(['EEGConformer', 'EEGNetv4', 'SepConv1DLite', 'ShallowFBCSPNet']),
                    'MAX_EPOCHS': np.random.randint(50, 200),  # 较少epochs快速筛选
                    'EARLY_STOPPING_PATIENCE': np.random.randint(10, 30),
                    'CV_FOLDS': np.random.choice([2, 3]),  # 较少CV
                    'CV_REPEATS': np.random.choice([1, 2]),
                    # 次要参数使用默认值
                    'WEIGHT_DECAY': 1e-4,
                    'NOISE_STD': 0.005,
                    'TIME_SHIFT_RANGE': 5,
                    'LABEL_SMOOTHING': 0.05,
                }
            elif stage == "fine":
                # 第二阶段：重要参数精调
                base_params = best_params if best_params else {}
                sample = {
                    'LEARNING_RATE': np.random.uniform(
                        max(0.001, base_params.get('LEARNING_RATE', 0.01) * 0.5),
                        min(0.1, base_params.get('LEARNING_RATE', 0.01) * 2.0)
                    ),
                    'BATCH_SIZE': np.random.choice([
                        max(32, base_params.get('BATCH_SIZE', 64) // 2),
                        base_params.get('BATCH_SIZE', 64),
                        min(256, base_params.get('BATCH_SIZE', 64) * 2)
                    ]),
                    'DROPOUT_RATE': np.random.uniform(
                        max(0.05, base_params.get('DROPOUT_RATE', 0.25) - 0.1),
                        min(0.7, base_params.get('DROPOUT_RATE', 0.25) + 0.1)
                    ),
                    'classifier': base_params.get('classifier', 'EEGConformer'),
                    'MAX_EPOCHS': np.random.randint(100, 300),
                    'EARLY_STOPPING_PATIENCE': np.random.randint(20, 50),
                    'CV_FOLDS': np.random.choice([3, 4, 5]),
                    'CV_REPEATS': np.random.choice([2, 3]),
                    # 开始调整次要参数
                    'WEIGHT_DECAY': np.random.uniform(1e-5, 1e-3),
                    'NOISE_STD': np.random.uniform(0.001, 0.01),
                    'TIME_SHIFT_RANGE': np.random.randint(3, 10),
                    'LABEL_SMOOTHING': np.random.uniform(0.0, 0.1),
                }
            elif stage == "final":
                # 第三阶段：所有参数精细调优
                base_params = best_params if best_params else {}
                sample = {
                    'LEARNING_RATE': np.random.uniform(
                        max(0.001, base_params.get('LEARNING_RATE', 0.01) * 0.8),
                        min(0.1, base_params.get('LEARNING_RATE', 0.01) * 1.2)
                    ),
                    'BATCH_SIZE': np.random.choice([
                        max(32, base_params.get('BATCH_SIZE', 64) - 32),
                        base_params.get('BATCH_SIZE', 64),
                        min(256, base_params.get('BATCH_SIZE', 64) + 32)
                    ]),
                    'DROPOUT_RATE': np.random.uniform(
                        max(0.05, base_params.get('DROPOUT_RATE', 0.25) - 0.05),
                        min(0.7, base_params.get('DROPOUT_RATE', 0.25) + 0.05)
                    ),
                    'classifier': base_params.get('classifier', 'EEGConformer'),
                    'MAX_EPOCHS': np.random.randint(200, 400),
                    'EARLY_STOPPING_PATIENCE': np.random.randint(30, 60),
                    'CV_FOLDS': np.random.choice([4, 5]),
                    'CV_REPEATS': np.random.choice([3, 4, 5]),
                    # 精细调整所有参数
                    'WEIGHT_DECAY': np.random.uniform(
                        max(1e-6, base_params.get('WEIGHT_DECAY', 1e-4) * 0.1),
                        min(1e-2, base_params.get('WEIGHT_DECAY', 1e-4) * 10)
                    ),
                    'NOISE_STD': np.random.uniform(
                        max(0.0001, base_params.get('NOISE_STD', 0.005) * 0.1),
                        min(0.05, base_params.get('NOISE_STD', 0.005) * 10)
                    ),
                    'TIME_SHIFT_RANGE': np.random.randint(
                        max(1, base_params.get('TIME_SHIFT_RANGE', 5) - 2),
                        min(20, base_params.get('TIME_SHIFT_RANGE', 5) + 2)
                    ),
                    'LABEL_SMOOTHING': np.random.uniform(
                        max(0.0, base_params.get('LABEL_SMOOTHING', 0.05) - 0.02),
                        min(0.2, base_params.get('LABEL_SMOOTHING', 0.05) + 0.02)
                    ),
                }
            
            samples.append(sample)
        
        return samples
    
    def get_gpu_status(self):
        """获取当前GPU状态"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                if len(gpu_info) >= 4:
                    util, mem_used, mem_total, temp = gpu_info
                    return f"GPU: {util}% | 内存: {mem_used}/{mem_total}MB | 温度: {temp}°C"
        except:
            pass
        return "GPU状态未知"
    
    def print_tuning_progress(self, stage: str, trial_idx: int, total_trials: int, 
                            best_accuracy: float, current_accuracy: float, 
                            elapsed_time: float, gpu_status: str):
        """打印实时调参进度"""
        progress_pct = (trial_idx + 1) / total_trials * 100
        elapsed_min = elapsed_time / 60
        
        print(f"\n{'='*80}")
        print(f"阶段: {stage} | 进度: {trial_idx + 1}/{total_trials} ({progress_pct:.1f}%)")
        print(f"当前准确率: {current_accuracy:.4f} | 最佳准确率: {best_accuracy:.4f}")
        print(f"已用时间: {elapsed_min:.1f}分钟 | {gpu_status}")
        print(f"{'='*80}")
    
    def analyze_parameter_importance(self, results: List[Dict]) -> Dict[str, float]:
        """分析参数重要性"""
        if len(results) < 5:
            return {}
        
        successful_results = [r for r in results if r['success']]
        if len(successful_results) < 3:
            return {}
        
        # 计算每个参数与准确率的相关性
        param_importance = {}
        param_names = ['LEARNING_RATE', 'BATCH_SIZE', 'DROPOUT_RATE', 'WEIGHT_DECAY', 
                      'NOISE_STD', 'TIME_SHIFT_RANGE', 'LABEL_SMOOTHING']
        
        for param in param_names:
            if param in successful_results[0]['params']:
                values = [r['params'][param] for r in successful_results]
                accuracies = [r['accuracy'] for r in successful_results]
                
                # 计算相关系数
                if len(set(values)) > 1:  # 确保参数有变化
                    correlation = np.corrcoef(values, accuracies)[0, 1]
                    param_importance[param] = abs(correlation) if not np.isnan(correlation) else 0
        
        return param_importance
    
    def run_layered_gpu_tuning(self, total_trials: int = 15):
        """运行分层GPU调参"""
        if not self.gpu_available:
            print("GPU不可用，无法运行GPU优化调参")
            return
        
        # 分层调参配置 - 确保每个阶段至少有1个试验
        if total_trials <= 3:
            # 试验数量少时，只使用一个阶段
            stage_config = {
                "coarse": {"trials": total_trials, "desc": "参数调优"}
            }
        else:
            # 正常分层配置
            coarse_trials = max(1, total_trials // 3)
            fine_trials = max(1, total_trials // 3)
            final_trials = total_trials - coarse_trials - fine_trials
            
            stage_config = {
                "coarse": {"trials": coarse_trials, "desc": "第一阶段：重要参数粗调"},
                "fine": {"trials": fine_trials, "desc": "第二阶段：重要参数精调"},
                "final": {"trials": final_trials, "desc": "第三阶段：所有参数精细调优"}
            }
        
        print(f"开始分层GPU调参: 总计 {total_trials} 个试验")
        print("="*80)
        
        all_results = []
        best_accuracy = -1.0
        best_params = None
        total_start_time = time.time()
        
        for stage, config in stage_config.items():
            if config["trials"] == 0:
                continue
                
            print(f"\n{config['desc']} - {config['trials']} 个试验")
            print("-" * 60)
            
            # 生成当前阶段的参数
            param_samples = self.generate_layered_params(stage, config["trials"], best_params)
            
            stage_start_time = time.time()
            stage_results = []
            
            # 创建阶段进度条
            with tqdm(total=config["trials"], desc=f"阶段: {stage}", unit="试验", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                for i, params in enumerate(param_samples):
                    trial_id = len(all_results) + i
                    
                    # 显示当前试验信息
                    pbar.write(f"开始试验 {trial_id + 1}: {params}")
                    
                    result = self.run_gpu_trial(params, trial_id, pbar)
                    stage_results.append(result)
                    
                    if result['success'] and result['accuracy'] > best_accuracy:
                        best_accuracy = result['accuracy']
                        best_params = result['params']
                        pbar.write(f"新最佳结果: {best_accuracy:.4f} (阶段: {stage})")
                        pbar.write(f"参数: {best_params}")
                    
                    # 实时保存结果
                    all_results.extend(stage_results)
                    self._save_results(all_results, best_params, best_accuracy)
                    
                    # 更新进度条
                    pbar.update(1)
                    pbar.refresh()  # 强制刷新进度条
                    
                    # 每3个试验显示一次GPU状态
                    if (i + 1) % 3 == 0:
                        gpu_status = self.get_gpu_status()
                        pbar.write(f"GPU状态: {gpu_status}")
            
            stage_time = time.time() - stage_start_time
            stage_best = max([r['accuracy'] for r in stage_results if r['success']], default=0.0)
            
            print(f"阶段 {stage} 完成:")
            print(f"  耗时: {stage_time/60:.1f}分钟")
            print(f"  最佳准确率: {stage_best:.4f}")
            print(f"  成功试验: {sum(1 for r in stage_results if r['success'])}/{len(stage_results)}")
            
            # 如果当前阶段没有找到更好的结果，提前结束
            if stage == "coarse" and stage_best < 0.5:
                print("第一阶段结果较差，建议检查数据或模型配置")
            elif stage == "fine" and stage_best <= best_accuracy:
                print("第二阶段未找到更好结果，跳过第三阶段")
                break
        
        total_time = time.time() - total_start_time
        
        # 保存最终结果
        self._save_results(all_results, best_params, best_accuracy, total_time)
        
        print(f"\n分层GPU调参完成！")
        print(f"总耗时: {total_time/60:.1f}分钟")
        print(f"最佳准确率: {best_accuracy:.4f}")
        print(f"成功试验: {sum(1 for r in all_results if r['success'])}/{len(all_results)}")
        print(f"最佳参数: {best_params}")
        
        return all_results, best_params, best_accuracy
    
    def run_gpu_optimized_tuning(self, n_trials: int = 15):
        """运行GPU优化调参（保持向后兼容）"""
        return self.run_layered_gpu_tuning(n_trials)
    
    def _save_results(self, results: List[Dict], best_params: Dict, best_accuracy: float, total_time: float = 0):
        """保存结果"""
        results_file = self.results_dir / "gpu_tuning_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'best_accuracy': best_accuracy,
                'best_params': best_params,
                'total_time': total_time,
                'gpu_available': self.gpu_available,
                'gpu_count': self.gpu_count,
                'gpu_memory_gb': self.gpu_memory_gb,
                'all_results': results,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        
        # 保存最佳参数配置
        if best_params:
            best_config_file = self.results_dir / "best_gpu_config.py"
            with open(self.base_config_path, 'r') as f:
                content = f.read()
            
            # 应用最佳参数
            modifications = {
                'DEVICE_MODE = \'auto\'': 'DEVICE_MODE = \'cuda\'',
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
    
    parser = argparse.ArgumentParser(description='GPU优化调参')
    parser.add_argument('--trials', type=int, default=15, help='试验数量')
    parser.add_argument('--mode', choices=['layered', 'random'], default='layered', 
                       help='调参模式: layered(分层调参) 或 random(随机搜索)')
    
    args = parser.parse_args()
    
    tuner = GPUOptimizedTuner()
    
    if args.mode == 'layered':
        print("使用分层调参模式")
        tuner.run_layered_gpu_tuning(args.trials)
    else:
        print("使用随机搜索模式")
        # 使用原始的随机搜索方法
        tuner.run_gpu_optimized_tuning(args.trials)

if __name__ == "__main__":
    main()
