#!/usr/bin/env python3
"""
分层调参策略 - 智能的多阶段超参数优化
1. 快速筛选阶段：少epochs快速筛选参数
2. 精细调优阶段：多epochs精细调优最佳参数
3. 最终验证阶段：完整训练验证最终结果
"""

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

class LayeredTuner:
    def __init__(self, base_config_path: str = "config.py"):
        self.base_config_path = base_config_path
        self.results_dir = Path("layered_tuning_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 分层调参配置
        self.stages = {
            'quick_screen': {
                'name': '快速筛选',
                'epochs': 50,
                'patience': 10,
                'cv_folds': 2,
                'cv_repeats': 1,
                'n_trials': 20,
                'description': '快速筛选最有希望的参数组合'
            },
            'fine_tune': {
                'name': '精细调优',
                'epochs': 150,
                'patience': 25,
                'cv_folds': 3,
                'cv_repeats': 2,
                'n_trials': 10,
                'description': '对筛选出的参数进行精细调优'
            },
            'final_validation': {
                'name': '最终验证',
                'epochs': 300,
                'patience': 50,
                'cv_folds': 5,
                'cv_repeats': 3,
                'n_trials': 3,
                'description': '最终验证最佳参数组合'
            }
        }
        
        self.all_results = {}
        self.best_params_history = []
        
    def create_stage_config(self, params: Dict[str, Any], stage_name: str, trial_id: int) -> str:
        """为特定阶段创建配置文件"""
        stage_config = self.stages[stage_name]
        
        # 读取基础配置
        with open(self.base_config_path, 'r') as f:
            content = f.read()
        
        # 修改关键参数
        modifications = {
            'MAX_EPOCHS = 500': f'MAX_EPOCHS = {stage_config["epochs"]}',
            'EARLY_STOPPING_PATIENCE = 50': f'EARLY_STOPPING_PATIENCE = {stage_config["patience"]}',
            'NESTED_CV_OUTER_FOLDS = 5': f'NESTED_CV_OUTER_FOLDS = {stage_config["cv_folds"]}',
            'NESTED_CV_REPEATS = 5': f'NESTED_CV_REPEATS = {stage_config["cv_repeats"]}',
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
        config_path = self.results_dir / f"config_{stage_name}_trial_{trial_id}.py"
        with open(config_path, 'w') as f:
            f.write(content)
        
        return str(config_path)
    
    def run_single_trial(self, params: Dict[str, Any], stage_name: str, trial_id: int) -> Dict[str, Any]:
        """运行单个试验"""
        stage_config = self.stages[stage_name]
        
        print(f"\n{'='*80}")
        print(f"阶段: {stage_config['name']} | 试验 {trial_id + 1}/{stage_config['n_trials']}")
        print(f"{'='*80}")
        print(f"配置: {stage_config['description']}")
        print(f"Epochs: {stage_config['epochs']} | CV: {stage_config['cv_folds']} folds, {stage_config['cv_repeats']} repeats")
        print(f"关键参数:")
        for key in ['LEARNING_RATE', 'classifier', 'BATCH_SIZE', 'DROPOUT_RATE']:
            if key in params:
                print(f"  {key}: {params[key]}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # 创建配置文件
            config_path = self.create_stage_config(params, stage_name, trial_id)
            
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
                timeout=3600  # 1小时超时
            )
            
            duration = time.time() - start_time
            
            # 清理配置文件
            if os.path.exists(config_path):
                os.unlink(config_path)
            
            if result.returncode != 0:
                print(f"试验失败 (返回码: {result.returncode})")
                print(f"错误: {result.stderr[:200]}...")
                return {
                    'success': False,
                    'accuracy': -1.0,
                    'error': result.stderr,
                    'duration': duration
                }
            
            # 提取准确率
            accuracy = self.extract_accuracy(result.stdout)
            
            print(f"试验完成 - 准确率: {accuracy:.4f} | 耗时: {duration/60:.1f}分钟")
            
            return {
                'success': True,
                'accuracy': accuracy,
                'duration': duration,
                'stdout': result.stdout[:500]
            }
            
        except subprocess.TimeoutExpired:
            print(f"试验超时")
            return {'success': False, 'accuracy': -1.0, 'error': 'timeout'}
        except Exception as e:
            print(f"试验异常: {e}")
            return {'success': False, 'accuracy': -1.0, 'error': str(e)}
    
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
    
    def generate_parameter_samples(self, stage_name: str, n_trials: int, previous_best: List[Dict] = None) -> List[Dict[str, Any]]:
        """生成参数样本"""
        if stage_name == 'quick_screen':
            # 快速筛选阶段：广泛搜索
            return self._generate_wide_search_samples(n_trials)
        elif stage_name == 'fine_tune' and previous_best:
            # 精细调优阶段：基于最佳参数进行局部搜索
            return self._generate_local_search_samples(previous_best, n_trials)
        elif stage_name == 'final_validation' and previous_best:
            # 最终验证阶段：使用最佳参数
            return [previous_best[0]] * n_trials
        else:
            return self._generate_wide_search_samples(n_trials)
    
    def _generate_wide_search_samples(self, n_trials: int) -> List[Dict[str, Any]]:
        """生成广泛搜索的参数样本"""
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
    
    def _generate_local_search_samples(self, best_params: List[Dict], n_trials: int) -> List[Dict[str, Any]]:
        """基于最佳参数生成局部搜索样本"""
        samples = []
        best = best_params[0]  # 使用最佳参数
        
        for _ in range(n_trials):
            sample = best.copy()
            
            # 对每个参数进行小幅扰动
            sample['LEARNING_RATE'] = max(0.001, best['LEARNING_RATE'] * np.random.uniform(0.5, 2.0))
            sample['BATCH_SIZE'] = np.random.choice([16, 24, 32, 48, 64])
            sample['DROPOUT_RATE'] = max(0.05, min(0.5, best['DROPOUT_RATE'] + np.random.uniform(-0.1, 0.1)))
            sample['WEIGHT_DECAY'] = max(1e-6, best['WEIGHT_DECAY'] * np.random.uniform(0.1, 10.0))
            sample['NOISE_STD'] = max(0.001, min(0.05, best['NOISE_STD'] + np.random.uniform(-0.005, 0.005)))
            sample['TIME_SHIFT_RANGE'] = max(1, min(20, best['TIME_SHIFT_RANGE'] + np.random.randint(-3, 4)))
            sample['LABEL_SMOOTHING'] = max(0.0, min(0.3, best['LABEL_SMOOTHING'] + np.random.uniform(-0.05, 0.05)))
            
            samples.append(sample)
        
        return samples
    
    def run_stage(self, stage_name: str) -> Dict[str, Any]:
        """运行单个阶段"""
        stage_config = self.stages[stage_name]
        print(f"\n{'='*100}")
        print(f"开始阶段: {stage_config['name']}")
        print(f"描述: {stage_config['description']}")
        print(f"配置: {stage_config['epochs']} epochs, {stage_config['cv_folds']} folds, {stage_config['cv_repeats']} repeats")
        print(f"试验数量: {stage_config['n_trials']}")
        print(f"{'='*100}")
        
        # 生成参数样本
        previous_best = self.best_params_history[-3:] if self.best_params_history else None
        param_samples = self.generate_parameter_samples(stage_name, stage_config['n_trials'], previous_best)
        
        stage_results = []
        best_accuracy = -1.0
        best_params = None
        
        for i, params in enumerate(param_samples):
            result = self.run_single_trial(params, stage_name, i)
            stage_results.append({
                'trial_id': i,
                'params': params,
                'result': result
            })
            
            if result['success'] and result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_params = params
                print(f"\n🎉 新最佳结果: {best_accuracy:.4f}")
                print(f"参数: {best_params}")
            
            # 实时保存结果
            self._save_stage_results(stage_name, stage_results, best_params, best_accuracy)
        
        # 保存最佳参数到历史记录
        if best_params:
            self.best_params_history.append(best_params)
        
        return {
            'stage_name': stage_name,
            'best_accuracy': best_accuracy,
            'best_params': best_params,
            'all_results': stage_results
        }
    
    def _save_stage_results(self, stage_name: str, results: List[Dict], best_params: Dict, best_accuracy: float):
        """保存阶段结果"""
        stage_file = self.results_dir / f"{stage_name}_results.json"
        with open(stage_file, 'w') as f:
            json.dump({
                'stage_name': stage_name,
                'best_accuracy': best_accuracy,
                'best_params': best_params,
                'all_results': results,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
    
    def run_layered_tuning(self):
        """运行完整的分层调参"""
        print("🚀 开始分层调参策略")
        print("="*100)
        
        start_time = time.time()
        
        # 运行各个阶段
        for stage_name in ['quick_screen', 'fine_tune', 'final_validation']:
            stage_result = self.run_stage(stage_name)
            self.all_results[stage_name] = stage_result
            
            print(f"\n✅ 阶段 '{stage_name}' 完成")
            print(f"最佳准确率: {stage_result['best_accuracy']:.4f}")
            print(f"最佳参数: {stage_result['best_params']}")
        
        total_time = time.time() - start_time
        
        # 生成最终报告
        self._generate_final_report(total_time)
        
        print(f"\n🎉 分层调参完成！总耗时: {total_time/3600:.1f}小时")
        print(f"最终最佳准确率: {self.all_results['final_validation']['best_accuracy']:.4f}")
    
    def _generate_final_report(self, total_time: float):
        """生成最终报告"""
        report = f"""
# 分层调参最终报告

## 总结
- 总耗时: {total_time/3600:.1f}小时
- 总试验数: {sum(len(self.all_results[stage]['all_results']) for stage in self.all_results)}

## 各阶段结果

"""
        
        for stage_name, result in self.all_results.items():
            stage_config = self.stages[stage_name]
            report += f"""
### {stage_config['name']}
- 最佳准确率: {result['best_accuracy']:.4f}
- 试验数量: {len(result['all_results'])}
- 配置: {stage_config['epochs']} epochs, {stage_config['cv_folds']} folds, {stage_config['cv_repeats']} repeats

最佳参数:
```json
{json.dumps(result['best_params'], indent=2)}
```

"""
        
        # 保存报告
        report_file = self.results_dir / "final_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"最终报告已保存到: {report_file}")

def main():
    """主函数"""
    tuner = LayeredTuner()
    tuner.run_layered_tuning()

if __name__ == "__main__":
    main()
