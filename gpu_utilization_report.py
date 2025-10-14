#!/usr/bin/env python3
"""
GPU利用率分析报告 - 分析当前调参代码的GPU使用情况
"""

import subprocess
import time
import os
import sys

def analyze_gpu_utilization():
    """分析GPU利用率"""
    print("🔍 GPU利用率分析报告")
    print("="*80)
    
    # 检查GPU配置
    print("📋 当前GPU配置:")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print(f"  {gpu_info}")
    except:
        print("  无法获取GPU信息")
    
    # 检查当前GPU状态
    print("\n📊 当前GPU状态:")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 5:
                util, mem_used, mem_total, temp, power = parts
                memory_percent = (int(mem_used) / int(mem_total)) * 100
                print(f"  使用率: {util}%")
                print(f"  内存使用: {mem_used}/{mem_total}MB ({memory_percent:.1f}%)")
                print(f"  温度: {temp}°C")
                print(f"  功耗: {power}W")
    except:
        print("  无法获取GPU状态")
    
    # 检查Python进程
    print("\n🐍 Python进程检查:")
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            python_processes = []
            for line in lines:
                if 'python' in line and ('main_tfdwt' in line or 'tune' in line):
                    python_processes.append(line)
            
            if python_processes:
                print(f"  发现 {len(python_processes)} 个相关Python进程:")
                for proc in python_processes:
                    print(f"    {proc}")
            else:
                print("  未发现相关Python进程")
    except:
        print("  无法检查进程状态")
    
    # 分析问题
    print("\n🔍 问题分析:")
    
    # 检查config.py中的设备设置
    try:
        with open('config.py', 'r') as f:
            content = f.read()
            if "DEVICE_MODE = 'cuda'" in content:
                print("  ✅ config.py已设置为使用CUDA")
            elif "DEVICE_MODE = 'auto'" in content:
                print("  ⚠️ config.py设置为自动选择设备")
            elif "DEVICE_MODE = 'cpu'" in content:
                print("  ❌ config.py设置为使用CPU")
            else:
                print("  ❓ 无法确定设备设置")
    except:
        print("  ❌ 无法读取config.py")
    
    # 检查CUDA可用性
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ PyTorch CUDA可用: {torch.cuda.device_count()}个GPU")
            print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
        else:
            print("  ❌ PyTorch CUDA不可用")
    except ImportError:
        print("  ❌ PyTorch未安装")
    
    # 建议
    print("\n💡 优化建议:")
    
    if util == "0" or int(util) < 10:
        print("  🚀 GPU使用率很低，建议:")
        print("    1. 增加batch_size (当前可能太小)")
        print("    2. 使用GPU优化调参脚本")
        print("    3. 检查模型是否正确移动到GPU")
        print("    4. 使用混合精度训练")
    
    if memory_percent < 20:
        print("  💾 GPU内存使用率低，建议:")
        print("    1. 增加batch_size到64-128")
        print("    2. 增加模型复杂度")
        print("    3. 使用更大的输入窗口")
    
    print("\n🎯 推荐操作:")
    print("  1. 运行GPU优化调参: python gpu_optimized_tuning.py --trials 10")
    print("  2. 监控GPU使用: python gpu_monitor.py --mode monitor --duration 300")
    print("  3. 检查训练日志: tail -f log_0909/TF_DWT_*.log")

def main():
    """主函数"""
    analyze_gpu_utilization()

if __name__ == "__main__":
    main()
