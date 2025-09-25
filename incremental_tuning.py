#!/usr/bin/env python3
"""
增量调参功能 - 基于已有结果继续调参，避免重复工作
"""

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

class IncrementalTuner:
    def __init__(self, base_config_path: str = "config.py", results_dir: str = "incremental_tuning_results"):
        self.base_config_path = base_config_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.results_file = self.results_dir / "tuning_history.json"
        self.tuning_history = self._load_history()
        
        print(f"增量调参初始化: 已有 {len(self.tuning_history.get('trials', []))} 个历史试验")
    
    def _load_history(self) -> Dict[str, Any]:
        """加载历史调参结果"""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载历史结果失败: {e}")
        
        return {
            'trials': [],
            'best_accuracy': -1.0,
            'best_params': None,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _save_history(self):
        """保存历史结果"""
        with open(self.results_file, 'w') as f:
            json.dump(self.tuning_history, f, indent=2)
    
    def analyze_history(self) -> Dict[str, Any]:
        """分析历史结果"""
        trials = self.tuning_history.get('trials', [])
        if not trials:
            return {'message': '没有历史试验数据'}
        
        # 统计信息
        successful_trials = [t for t in trials if t.get('success', False)]
        accuracies = [t['accuracy'] for t in successful_trials]
        
        if not accuracies:
            return {'message': '没有成功的试验'}
        
        # 参数分析
        param_analysis = {}
        for param_name in ['LEARNING_RATE', 'BATCH_SIZE', 'DROPOUT_RATE', 'classifier']:
            param_values = {}
            for trial in successful_trials:
                if param_name in trial.get('params', {}):
                    value = trial['params'][param_name]
                    if value not in param_values:
                        param_values[value] = []
                    param_values[value].append(trial['accuracy'])
            
            # 计算每个参数值的平均准确率
            param_analysis[param_name] = {}
            for value, accs in param_values.items():
                param_analysis[param_name][value] = {
                    'mean_accuracy': np.mean(accs),
                    'count': len(accs),
                    'max_accuracy': max(accs)
                }
        
        return {
            'total_trials': len(trials),
            'successful_trials': len(successful_trials),
            'best_accuracy': max(accuracies),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'param_analysis': param_analysis,
            'top_5_trials': sorted(successful_trials, key=lambda x: x['accuracy'], reverse=True)[:5]
        }
    
    def generate_smart_samples(self, n_trials: int, strategy: str = 'exploit_explore') -> List[Dict[str, Any]]:
        """基于历史结果生成智能参数样本"""
        analysis = self.analyze_history()
        
        if 'message' in analysis:
            # 没有历史数据，使用随机搜索
            return self._generate_random_samples(n_trials)
        
        samples = []
        
        if strategy == 'exploit':
            # 纯利用策略：基于最佳参数进行局部搜索
            samples = self._generate_exploit_samples(analysis, n_trials)
        elif strategy == 'explore':
            # 纯探索策略：探索未充分测试的参数空间
            samples = self._generate_explore_samples(analysis, n_trials)
        else:  # exploit_explore
            # 平衡策略：70%利用，30%探索
            exploit_count = int(n_trials * 0.7)
            explore_count = n_trials - exploit_count
            
            samples.extend(self._generate_exploit_samples(analysis, exploit_count))
            samples.extend(self._generate_explore_samples(analysis, explore_count))
        
        return samples
    
    def _generate_random_samples(self, n_trials: int) -> List[Dict[str, Any]]:
        """生成随机参数样本"""
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
            }
            samples.append(sample)
        return samples
    
    def _generate_exploit_samples(self, analysis: Dict[str, Any], n_trials: int) -> List[Dict[str, Any]]:
        """基于最佳参数生成利用样本"""
        samples = []
        best_trial = analysis['top_5_trials'][0]
        best_params = best_trial['params']
        
        for _ in range(n_trials):
            sample = best_params.copy()
            
            # 对每个参数进行小幅扰动
            sample['LEARNING_RATE'] = max(0.001, best_params['LEARNING_RATE'] * np.random.uniform(0.5, 2.0))
            sample['BATCH_SIZE'] = np.random.choice([16, 24, 32, 48, 64])
            sample['DROPOUT_RATE'] = max(0.05, min(0.5, best_params['DROPOUT_RATE'] + np.random.uniform(-0.1, 0.1)))
            sample['WEIGHT_DECAY'] = max(1e-6, best_params['WEIGHT_DECAY'] * np.random.uniform(0.1, 10.0))
            sample['NOISE_STD'] = max(0.001, min(0.05, best_params['NOISE_STD'] + np.random.uniform(-0.005, 0.005)))
            sample['TIME_SHIFT_RANGE'] = max(1, min(20, best_params['TIME_SHIFT_RANGE'] + np.random.randint(-3, 4)))
            sample['LABEL_SMOOTHING'] = max(0.0, min(0.3, best_params['LABEL_SMOOTHING'] + np.random.uniform(-0.05, 0.05)))
            
            samples.append(sample)
        
        return samples
    
    def _generate_explore_samples(self, analysis: Dict[str, Any], n_trials: int) -> List[Dict[str, Any]]:
        """生成探索样本"""
        samples = []
        param_analysis = analysis['param_analysis']
        
        for _ in range(n_trials):
            sample = {}
            
            # 基于历史分析选择参数
            for param_name in ['LEARNING_RATE', 'BATCH_SIZE', 'DROPOUT_RATE', 'classifier']:
                if param_name in param_analysis:
                    # 选择表现较差的参数值进行探索
                    param_data = param_analysis[param_name]
                    if param_name == 'classifier':
                        # 分类器选择
                        sample[param_name] = np.random.choice(['EEGConformer', 'EEGNetv4', 'SepConv1DLite', 'ShallowFBCSPNet'])
                    elif param_name == 'BATCH_SIZE':
                        # 批次大小选择
                        sample[param_name] = np.random.choice([16, 24, 32, 48, 64])
                    else:
                        # 连续参数：在未充分测试的范围内采样
                        if param_name == 'LEARNING_RATE':
                            sample[param_name] = np.random.uniform(0.001, 0.05)
                        elif param_name == 'DROPOUT_RATE':
                            sample[param_name] = np.random.uniform(0.1, 0.4)
                        elif param_name == 'WEIGHT_DECAY':
                            sample[param_name] = np.random.uniform(1e-5, 1e-2)
                        elif param_name == 'NOISE_STD':
                            sample[param_name] = np.random.uniform(0.001, 0.02)
                        elif param_name == 'TIME_SHIFT_RANGE':
                            sample[param_name] = np.random.randint(2, 15)
                        elif param_name == 'LABEL_SMOOTHING':
                            sample[param_name] = np.random.uniform(0.0, 0.2)
                else:
                    # 使用默认值
                    sample[param_name] = self._get_default_param(param_name)
            
            samples.append(sample)
        
        return samples
    
    def _get_default_param(self, param_name: str) -> Any:
        """获取参数默认值"""
        defaults = {
            'LEARNING_RATE': 0.01,
            'BATCH_SIZE': 32,
            'DROPOUT_RATE': 0.25,
            'WEIGHT_DECAY': 1e-4,
            'classifier': 'EEGConformer',
            'NOISE_STD': 0.005,
            'TIME_SHIFT_RANGE': 5,
            'LABEL_SMOOTHING': 0.05,
        }
        return defaults.get(param_name, 0.01)
    
    def create_config_for_trial(self, params: Dict[str, Any], trial_id: int) -> str:
        """创建试验配置文件"""
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
        config_path = self.results_dir / f"config_trial_{trial_id}.py"
        with open(config_path, 'w') as f:
            f.write(content)
        
        return str(config_path)
    
    def run_single_trial(self, params: Dict[str, Any], trial_id: int) -> Dict[str, Any]:
        """运行单个试验"""
        print(f"\n试验 {trial_id + 1} 开始")
        print(f"参数: {params}")
        
        start_time = time.time()
        
        try:
            # 创建配置文件
            config_path = self.create_config_for_trial(params, trial_id)
            
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
            
            duration = time.time() - start_time
            
            # 清理配置文件
            if os.path.exists(config_path):
                os.unlink(config_path)
            
            if result.returncode != 0:
                print(f"试验 {trial_id + 1} 失败")
                return {
                    'trial_id': trial_id,
                    'success': False,
                    'accuracy': -1.0,
                    'error': result.stderr[:200],
                    'params': params,
                    'duration': duration
                }
            
            # 提取准确率
            accuracy = self.extract_accuracy(result.stdout)
            
            print(f"试验 {trial_id + 1} 完成 - 准确率: {accuracy:.4f}")
            
            return {
                'trial_id': trial_id,
                'success': True,
                'accuracy': accuracy,
                'params': params,
                'duration': duration,
                'stdout': result.stdout[:300]
            }
            
        except subprocess.TimeoutExpired:
            print(f"试验 {trial_id + 1} 超时")
            return {
                'trial_id': trial_id,
                'success': False,
                'accuracy': -1.0,
                'error': 'timeout',
                'params': params
            }
        except Exception as e:
            print(f"试验 {trial_id + 1} 异常: {e}")
            return {
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
    
    def run_incremental_tuning(self, n_trials: int = 10, strategy: str = 'exploit_explore'):
        """运行增量调参"""
        print(f"🚀 开始增量调参: {n_trials} 个试验, 策略: {strategy}")
        print("="*80)
        
        # 显示历史分析
        analysis = self.analyze_history()
        if 'message' not in analysis:
            print(f"历史最佳准确率: {analysis['best_accuracy']:.4f}")
            print(f"历史平均准确率: {analysis['mean_accuracy']:.4f}")
            print(f"历史试验总数: {analysis['total_trials']}")
        
        # 生成智能参数样本
        param_samples = self.generate_smart_samples(n_trials, strategy)
        
        # 运行试验
        start_time = time.time()
        best_accuracy = self.tuning_history.get('best_accuracy', -1.0)
        best_params = self.tuning_history.get('best_params')
        
        for i, params in enumerate(param_samples):
            result = self.run_single_trial(params, i)
            
            # 添加到历史记录
            self.tuning_history['trials'].append(result)
            
            # 更新最佳结果
            if result['success'] and result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_params = result['params']
                print(f"\n🎉 新最佳结果: {best_accuracy:.4f}")
                print(f"参数: {best_params}")
            
            # 实时保存
            self.tuning_history['best_accuracy'] = best_accuracy
            self.tuning_history['best_params'] = best_params
            self.tuning_history['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
            self._save_history()
            
            # 进度报告
            completed = i + 1
            elapsed_time = time.time() - start_time
            avg_time_per_trial = elapsed_time / completed
            remaining_trials = n_trials - completed
            estimated_remaining_time = remaining_trials * avg_time_per_trial
            
            print(f"进度: {completed}/{n_trials} ({completed/n_trials*100:.1f}%) | "
                  f"最佳: {best_accuracy:.4f} | "
                  f"预计剩余: {estimated_remaining_time/60:.1f}分钟")
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 增量调参完成！")
        print(f"本次耗时: {total_time/60:.1f}分钟")
        print(f"历史最佳准确率: {best_accuracy:.4f}")
        print(f"历史试验总数: {len(self.tuning_history['trials'])}")
        
        return best_params, best_accuracy

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='增量调参')
    parser.add_argument('--trials', type=int, default=10, help='试验数量')
    parser.add_argument('--strategy', choices=['exploit', 'explore', 'exploit_explore'], 
                       default='exploit_explore', help='调参策略')
    parser.add_argument('--analyze', action='store_true', help='只分析历史结果')
    
    args = parser.parse_args()
    
    tuner = IncrementalTuner()
    
    if args.analyze:
        # 只分析历史结果
        analysis = tuner.analyze_history()
        print("历史结果分析:")
        print(json.dumps(analysis, indent=2))
    else:
        # 运行增量调参
        tuner.run_incremental_tuning(args.trials, args.strategy)

if __name__ == "__main__":
    main()
