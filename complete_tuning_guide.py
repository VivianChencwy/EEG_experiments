#!/usr/bin/env python3
"""
完整的调参运行指南 - 集成所有调参策略
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any

class CompleteTuningGuide:
    def __init__(self):
        self.available_strategies = {
            'quick': {
                'name': '快速验证',
                'description': '3个试验，极低epochs，验证代码逻辑',
                'command': 'python quick_validation_test.py',
                'estimated_time': '5-10分钟',
                'use_case': '首次运行，验证系统可用性'
            },
            'layered': {
                'name': '分层调参',
                'description': '智能多阶段调参：快速筛选→精细调优→最终验证',
                'command': 'python layered_tuning.py',
                'estimated_time': '2-4小时',
                'use_case': '推荐的主要调参策略'
            },
            'parallel': {
                'name': '并行调参',
                'description': '多进程并行调参，充分利用计算资源',
                'command': 'python parallel_tuning.py --trials 20 --workers 4',
                'estimated_time': '1-2小时',
                'use_case': '有多核CPU或GPU资源时使用'
            },
            'incremental': {
                'name': '增量调参',
                'description': '基于历史结果继续调参，避免重复工作',
                'command': 'python incremental_tuning.py --trials 10 --strategy exploit_explore',
                'estimated_time': '30-60分钟',
                'use_case': '已有调参历史，继续优化'
            },
            'standard': {
                'name': '标准调参',
                'description': '使用原始调参脚本的标准模式',
                'command': 'python run_tuning_example.py --mode standard',
                'estimated_time': '12-25小时',
                'use_case': '传统调参方法，需要长时间运行'
            }
        }
    
    def print_guide(self):
        """打印完整的调参指南"""
        print("="*100)
        print("🚀 TFDWT 完整调参运行指南")
        print("="*100)
        
        print("\n📋 调参策略概览:")
        for strategy, info in self.available_strategies.items():
            print(f"\n{strategy.upper()}: {info['name']}")
            print(f"  描述: {info['description']}")
            print(f"  命令: {info['command']}")
            print(f"  预计时间: {info['estimated_time']}")
            print(f"  适用场景: {info['use_case']}")
        
        print("\n" + "="*100)
        print("🎯 推荐运行顺序")
        print("="*100)
        
        print("\n1️⃣ 首次运行 - 验证系统可用性")
        print("   python quick_validation_test.py")
        print("   ⏱️  预计时间: 5-10分钟")
        print("   ✅ 验证代码逻辑正确，系统可用")
        
        print("\n2️⃣ 主要调参 - 分层调参策略（推荐）")
        print("   python layered_tuning.py")
        print("   ⏱️  预计时间: 2-4小时")
        print("   🎯 智能多阶段调参，效率最高")
        
        print("\n3️⃣ 资源充足时 - 并行调参")
        print("   python parallel_tuning.py --trials 20 --workers 4")
        print("   ⏱️  预计时间: 1-2小时")
        print("   💪 充分利用多核资源")
        
        print("\n4️⃣ 继续优化 - 增量调参")
        print("   python incremental_tuning.py --trials 10 --strategy exploit_explore")
        print("   ⏱️  预计时间: 30-60分钟")
        print("   🔄 基于历史结果继续优化")
        
        print("\n" + "="*100)
        print("📊 实时监控命令")
        print("="*100)
        
        print("\n监控调参进度:")
        print("   # 查看最新日志")
        print("   tail -f log_0909/TF_DWT_*.log")
        print("")
        print("   # 查看进程状态")
        print("   ps aux | grep python | grep -E '(tune|main_tfdwt)'")
        print("")
        print("   # 查看GPU使用情况")
        print("   nvidia-smi -l 1")
        print("")
        print("   # 查看磁盘空间")
        print("   df -h")
        
        print("\n" + "="*100)
        print("🔧 调参参数说明")
        print("="*100)
        
        print("\n关键参数范围:")
        print("   LEARNING_RATE: 0.001 - 0.05 (学习率)")
        print("   BATCH_SIZE: 16, 24, 32, 48, 64 (批次大小)")
        print("   DROPOUT_RATE: 0.1 - 0.4 (Dropout率)")
        print("   WEIGHT_DECAY: 1e-6 - 1e-2 (权重衰减)")
        print("   classifier: EEGConformer, EEGNetv4, SepConv1DLite, ShallowFBCSPNet")
        print("   NOISE_STD: 0.001 - 0.02 (数据增强噪声)")
        print("   TIME_SHIFT_RANGE: 2 - 15 (时间偏移范围)")
        print("   LABEL_SMOOTHING: 0.0 - 0.2 (标签平滑)")
        
        print("\n" + "="*100)
        print("📁 结果文件说明")
        print("="*100)
        
        print("\n调参结果文件:")
        print("   layered_tuning_results/     - 分层调参结果")
        print("   parallel_tuning_results/    - 并行调参结果")
        print("   incremental_tuning_results/ - 增量调参结果")
        print("   quick_test_results/         - 快速测试结果")
        print("   tfdwt_*results*.csv         - 详细结果CSV")
        print("   log_0909/TF_DWT_*.log       - 训练日志")
        
        print("\n" + "="*100)
        print("⚠️  注意事项")
        print("="*100)
        
        print("\n1. 环境准备:")
        print("   conda activate eeg_realtime")
        print("   cd /home/vivian/eeg/EEG_experiments")
        
        print("\n2. 资源要求:")
        print("   - 内存: 至少8GB RAM")
        print("   - 存储: 至少10GB可用空间")
        print("   - GPU: 可选，但会显著加速")
        
        print("\n3. 中断恢复:")
        print("   - 所有调参脚本都支持中断后继续")
        print("   - 结果会实时保存，不会丢失")
        print("   - 使用Ctrl+C安全中断")
        
        print("\n" + "="*100)
        print("🎉 开始调参！")
        print("="*100)
    
    def run_strategy(self, strategy: str, **kwargs):
        """运行指定的调参策略"""
        if strategy not in self.available_strategies:
            print(f"❌ 未知策略: {strategy}")
            print(f"可用策略: {list(self.available_strategies.keys())}")
            return
        
        info = self.available_strategies[strategy]
        print(f"🚀 开始运行: {info['name']}")
        print(f"描述: {info['description']}")
        print(f"预计时间: {info['estimated_time']}")
        print("="*80)
        
        # 构建命令
        command = info['command']
        if kwargs:
            # 添加额外参数
            for key, value in kwargs.items():
                command += f" --{key} {value}"
        
        print(f"执行命令: {command}")
        print("="*80)
        
        # 执行命令
        import subprocess
        try:
            result = subprocess.run(command, shell=True, check=True)
            print(f"✅ {info['name']} 完成！")
        except subprocess.CalledProcessError as e:
            print(f"❌ {info['name']} 失败: {e}")
        except KeyboardInterrupt:
            print(f"⏹️  {info['name']} 被用户中断")
    
    def check_system_status(self):
        """检查系统状态"""
        print("🔍 系统状态检查")
        print("="*50)
        
        # 检查必要文件
        required_files = [
            'main_tfdwt.py',
            'config.py',
            'layered_tuning.py',
            'parallel_tuning.py',
            'incremental_tuning.py',
            'quick_validation_test.py'
        ]
        
        print("📁 必要文件检查:")
        for file in required_files:
            if Path(file).exists():
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file} - 缺失")
        
        # 检查环境
        print("\n🐍 Python环境检查:")
        print(f"   Python版本: {sys.version}")
        print(f"   工作目录: {os.getcwd()}")
        
        # 检查GPU
        print("\n🖥️  GPU检查:")
        try:
            import torch
            if torch.cuda.is_available():
                print(f"   ✅ CUDA可用: {torch.cuda.get_device_name(0)}")
                print(f"   GPU数量: {torch.cuda.device_count()}")
            else:
                print("   ⚠️  CUDA不可用，将使用CPU")
        except ImportError:
            print("   ❌ PyTorch未安装")
        
        # 检查内存
        print("\n💾 内存检查:")
        try:
            import psutil
            memory = psutil.virtual_memory()
            print(f"   总内存: {memory.total / (1024**3):.1f} GB")
            print(f"   可用内存: {memory.available / (1024**3):.1f} GB")
            print(f"   使用率: {memory.percent:.1f}%")
        except ImportError:
            print("   ⚠️  无法检查内存状态")
        
        print("\n" + "="*50)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='完整调参指南')
    parser.add_argument('--strategy', choices=['quick', 'layered', 'parallel', 'incremental', 'standard'],
                       help='运行指定策略')
    parser.add_argument('--guide', action='store_true', help='显示完整指南')
    parser.add_argument('--check', action='store_true', help='检查系统状态')
    parser.add_argument('--trials', type=int, help='试验数量')
    parser.add_argument('--workers', type=int, help='工作进程数')
    
    args = parser.parse_args()
    
    guide = CompleteTuningGuide()
    
    if args.check:
        guide.check_system_status()
    elif args.strategy:
        kwargs = {}
        if args.trials:
            kwargs['trials'] = args.trials
        if args.workers:
            kwargs['workers'] = args.workers
        guide.run_strategy(args.strategy, **kwargs)
    else:
        guide.print_guide()

if __name__ == "__main__":
    main()
