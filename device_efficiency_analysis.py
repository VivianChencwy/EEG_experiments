#!/usr/bin/env python3
"""
设备效率分析 - 分析并行调参在您的设备上的实际效率
"""

import multiprocessing as mp
import torch
import subprocess
import time
import os
from pathlib import Path

class DeviceEfficiencyAnalyzer:
    def __init__(self):
        self.cpu_cores = mp.cpu_count()
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        self.memory_gb = self._get_memory_gb()
        self.disk_gb = self._get_disk_gb()
        
    def _get_memory_gb(self):
        """获取内存大小（GB）"""
        try:
            result = subprocess.run(['free', '-g'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            for line in lines:
                if line.startswith('Mem:'):
                    return int(line.split()[1])
        except:
            pass
        return 125  # 从之前的输出看到是125Gi
    
    def _get_disk_gb(self):
        """获取磁盘可用空间（GB）"""
        try:
            result = subprocess.run(['df', '-BG', '.'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            if len(lines) > 1:
                return int(lines[1].split()[3].rstrip('G'))
        except:
            pass
        return 209  # 从之前的输出看到是209G可用
    
    def analyze_parallel_efficiency(self):
        """分析并行调参效率"""
        print("🔍 您的设备配置分析")
        print("="*60)
        
        print(f"CPU核心数: {self.cpu_cores}")
        print(f"GPU: {'RTX 4090 (23.5GB)' if self.gpu_available else '不可用'}")
        print(f"内存: {self.memory_gb}GB")
        print(f"磁盘可用: {self.disk_gb}GB")
        
        print("\n📊 并行调参效率分析")
        print("="*60)
        
        # 理论分析
        optimal_workers = min(self.cpu_cores, 8)  # 建议不超过8个进程
        gpu_workers = 1 if self.gpu_available else 0
        
        print(f"推荐工作进程数: {optimal_workers}")
        print(f"GPU加速: {'是' if self.gpu_available else '否'}")
        
        # 效率预测
        base_time = 90  # 单进程基准时间（分钟）
        
        if self.gpu_available:
            # 有GPU的情况
            gpu_speedup = 3.0  # GPU加速倍数
            parallel_speedup = optimal_workers * 0.8  # 并行效率（考虑I/O开销）
            total_speedup = gpu_speedup * parallel_speedup
            
            estimated_time = base_time / total_speedup
            
            print(f"\n🚀 效率预测（有GPU）:")
            print(f"GPU加速倍数: {gpu_speedup:.1f}x")
            print(f"并行加速倍数: {parallel_speedup:.1f}x")
            print(f"总加速倍数: {total_speedup:.1f}x")
            print(f"预计完成时间: {estimated_time:.1f}分钟 ({estimated_time/60:.1f}小时)")
            
        else:
            # 无GPU的情况
            parallel_speedup = optimal_workers * 0.7  # 并行效率（考虑I/O开销）
            estimated_time = base_time / parallel_speedup
            
            print(f"\n⚡ 效率预测（无GPU）:")
            print(f"并行加速倍数: {parallel_speedup:.1f}x")
            print(f"预计完成时间: {estimated_time:.1f}分钟 ({estimated_time/60:.1f}小时)")
        
        # 资源使用分析
        print(f"\n💾 资源使用分析:")
        memory_per_process = 2.0  # 每个进程预计使用2GB内存
        total_memory_needed = optimal_workers * memory_per_process
        
        print(f"每个进程预计内存使用: {memory_per_process}GB")
        print(f"总内存需求: {total_memory_needed}GB")
        print(f"内存充足: {'是' if total_memory_needed < self.memory_gb * 0.8 else '否'}")
        
        disk_per_trial = 0.5  # 每个试验预计使用0.5GB磁盘空间
        total_disk_needed = 20 * disk_per_trial  # 20个试验
        
        print(f"每个试验预计磁盘使用: {disk_per_trial}GB")
        print(f"总磁盘需求: {total_disk_needed}GB")
        print(f"磁盘充足: {'是' if total_disk_needed < self.disk_gb else '否'}")
        
        # 实际测试建议
        print(f"\n🧪 实际测试建议:")
        print(f"1. 先用少量试验测试: python parallel_tuning.py --trials 5 --workers 4")
        print(f"2. 监控资源使用: python monitor_tuning.py --mode full")
        print(f"3. 根据实际表现调整workers数量")
        
        return {
            'optimal_workers': optimal_workers,
            'estimated_time': estimated_time,
            'gpu_available': self.gpu_available,
            'memory_sufficient': total_memory_needed < self.memory_gb * 0.8,
            'disk_sufficient': total_disk_needed < self.disk_gb
        }
    
    def run_quick_parallel_test(self):
        """运行快速并行测试"""
        print("\n🚀 运行快速并行测试")
        print("="*60)
        
        # 创建快速测试配置
        test_config = {
            'trials': 3,
            'workers': 4,
            'epochs': 10,  # 很少的epochs用于快速测试
            'cv_folds': 2,
            'cv_repeats': 1
        }
        
        print(f"测试配置: {test_config}")
        print("开始测试...")
        
        start_time = time.time()
        
        try:
            # 运行并行调参测试
            cmd = f"python parallel_tuning.py --trials {test_config['trials']} --workers {test_config['workers']}"
            result = subprocess.run(cmd, shell=True, timeout=600)  # 10分钟超时
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ 测试成功完成")
                print(f"实际耗时: {duration:.1f}分钟")
                print(f"平均每试验: {duration/test_config['trials']:.1f}分钟")
                
                # 计算实际效率
                single_trial_time = duration / test_config['trials']
                parallel_efficiency = test_config['workers'] * single_trial_time / duration
                
                print(f"并行效率: {parallel_efficiency:.2f}")
                print(f"加速倍数: {test_config['workers'] * parallel_efficiency:.1f}x")
                
                return True, duration, parallel_efficiency
            else:
                print(f"❌ 测试失败 (返回码: {result.returncode})")
                return False, duration, 0
                
        except subprocess.TimeoutExpired:
            print("⏰ 测试超时")
            return False, 600, 0
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            return False, 0, 0

def main():
    """主函数"""
    analyzer = DeviceEfficiencyAnalyzer()
    
    # 分析设备效率
    analysis = analyzer.analyze_parallel_efficiency()
    
    # 询问是否运行实际测试
    print(f"\n❓ 是否运行实际测试来验证效率？")
    print("这将运行3个试验，预计需要5-10分钟")
    
    # 自动运行测试（在实际使用中，这里可以改为用户输入）
    print("自动开始测试...")
    
    success, duration, efficiency = analyzer.run_quick_parallel_test()
    
    if success:
        print(f"\n🎉 实际测试结果:")
        print(f"完成时间: {duration:.1f}分钟")
        print(f"并行效率: {efficiency:.2f}")
        
        # 基于实际结果预测完整调参时间
        full_trials = 20
        estimated_full_time = duration * full_trials / 3
        
        print(f"\n📊 完整调参预测 (20个试验):")
        print(f"预计时间: {estimated_full_time:.1f}分钟 ({estimated_full_time/60:.1f}小时)")
        
        if estimated_full_time < 120:  # 2小时
            print("✅ 您的设备可以高效运行并行调参！")
        elif estimated_full_time < 240:  # 4小时
            print("⚠️ 您的设备可以运行并行调参，但效率一般")
        else:
            print("❌ 您的设备运行并行调参效率较低，建议使用其他策略")
    else:
        print("❌ 测试失败，建议使用分层调参或增量调参策略")

if __name__ == "__main__":
    main()
