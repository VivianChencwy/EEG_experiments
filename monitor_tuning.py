#!/usr/bin/env python3
"""
实时监控调参进度 - 提供实时输出证明代码正在正常运行
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import threading
import queue

class TuningMonitor:
    def __init__(self):
        self.monitoring = False
        self.log_queue = queue.Queue()
        
    def monitor_logs(self, log_pattern: str = "log_0909/TF_DWT_*.log"):
        """监控日志文件"""
        print("📊 开始监控训练日志...")
        print("="*80)
        
        try:
            # 找到最新的日志文件
            log_files = list(Path(".").glob(log_pattern))
            if not log_files:
                print("❌ 未找到日志文件")
                return
            
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            print(f"📁 监控文件: {latest_log}")
            print("="*80)
            
            # 使用tail -f监控日志
            process = subprocess.Popen(
                ['tail', '-f', str(latest_log)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            while self.monitoring:
                line = process.stdout.readline()
                if line:
                    # 高亮重要信息
                    if "Epoch" in line and "LR=" in line:
                        print(f"🔄 {line.strip()}")
                    elif "Epoch Summary" in line:
                        print(f"📈 {line.strip()}")
                    elif "New best" in line:
                        print(f"🎉 {line.strip()}")
                    elif "Early stopping" in line:
                        print(f"⏹️  {line.strip()}")
                    elif "Overall accuracy" in line:
                        print(f"🎯 {line.strip()}")
                    elif "CROSS-VALIDATION RESULTS" in line:
                        print(f"📊 {line.strip()}")
                    else:
                        print(f"   {line.strip()}")
                else:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n⏹️  监控已停止")
        except Exception as e:
            print(f"❌ 监控错误: {e}")
        finally:
            if 'process' in locals():
                process.terminate()
    
    def monitor_processes(self):
        """监控相关进程"""
        print("🔍 进程监控:")
        print("="*50)
        
        try:
            # 查找相关进程
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True,
                text=True
            )
            
            lines = result.stdout.split('\n')
            tuning_processes = []
            
            for line in lines:
                if any(keyword in line for keyword in ['tune', 'main_tfdwt', 'layered', 'parallel', 'incremental']):
                    tuning_processes.append(line)
            
            if tuning_processes:
                print("✅ 发现调参进程:")
                for proc in tuning_processes:
                    print(f"   {proc}")
            else:
                print("❌ 未发现调参进程")
                
        except Exception as e:
            print(f"❌ 进程监控错误: {e}")
    
    def monitor_gpu(self):
        """监控GPU使用情况"""
        print("🖥️  GPU监控:")
        print("="*50)
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(result.stdout)
            else:
                print("❌ GPU不可用或nvidia-smi未安装")
                
        except Exception as e:
            print(f"❌ GPU监控错误: {e}")
    
    def monitor_results(self):
        """监控结果文件"""
        print("📁 结果文件监控:")
        print("="*50)
        
        result_dirs = [
            "layered_tuning_results",
            "parallel_tuning_results", 
            "incremental_tuning_results",
            "quick_test_results"
        ]
        
        for dir_name in result_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                print(f"✅ {dir_name}/")
                
                # 列出文件
                files = list(dir_path.glob("*"))
                for file in files:
                    if file.is_file():
                        size = file.stat().st_size
                        mtime = time.ctime(file.stat().st_mtime)
                        print(f"   📄 {file.name} ({size} bytes, {mtime})")
            else:
                print(f"❌ {dir_name}/ - 不存在")
    
    def monitor_system_resources(self):
        """监控系统资源"""
        print("💾 系统资源监控:")
        print("="*50)
        
        try:
            # 内存使用
            result = subprocess.run(['free', '-h'], capture_output=True, text=True)
            if result.returncode == 0:
                print("内存使用:")
                for line in result.stdout.split('\n'):
                    if 'Mem:' in line or 'Swap:' in line:
                        print(f"   {line}")
            
            # 磁盘使用
            result = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
            if result.returncode == 0:
                print("\n磁盘使用:")
                lines = result.stdout.split('\n')
                if len(lines) > 1:
                    print(f"   {lines[1]}")
                    
        except Exception as e:
            print(f"❌ 系统资源监控错误: {e}")
    
    def start_monitoring(self, log_pattern: str = "log_0909/TF_DWT_*.log"):
        """开始全面监控"""
        self.monitoring = True
        
        print("🚀 开始全面监控调参进度")
        print("="*100)
        
        # 启动日志监控线程
        log_thread = threading.Thread(target=self.monitor_logs, args=(log_pattern,))
        log_thread.daemon = True
        log_thread.start()
        
        try:
            while self.monitoring:
                # 每30秒更新一次系统状态
                time.sleep(30)
                
                print("\n" + "="*100)
                print(f"📊 系统状态更新 - {time.strftime('%H:%M:%S')}")
                print("="*100)
                
                self.monitor_processes()
                print()
                self.monitor_gpu()
                print()
                self.monitor_results()
                print()
                self.monitor_system_resources()
                
        except KeyboardInterrupt:
            print("\n⏹️  监控已停止")
        finally:
            self.monitoring = False
    
    def quick_status(self):
        """快速状态检查"""
        print("⚡ 快速状态检查")
        print("="*50)
        
        # 检查进程
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'python.*tune'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("✅ 调参进程正在运行")
            else:
                print("❌ 未发现调参进程")
        except:
            print("❌ 无法检查进程状态")
        
        # 检查最新日志
        log_files = list(Path(".").glob("log_0909/TF_DWT_*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            mtime = time.ctime(latest_log.stat().st_mtime)
            print(f"📁 最新日志: {latest_log.name} ({mtime})")
            
            # 显示最后几行
            try:
                result = subprocess.run(
                    ['tail', '-5', str(latest_log)],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print("最新日志内容:")
                    for line in result.stdout.split('\n'):
                        if line.strip():
                            print(f"   {line.strip()}")
            except:
                pass
        else:
            print("❌ 未找到日志文件")
        
        # 检查结果文件
        result_dirs = ["layered_tuning_results", "parallel_tuning_results", "incremental_tuning_results"]
        for dir_name in result_dirs:
            if Path(dir_name).exists():
                print(f"✅ {dir_name}/ 存在")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='调参进度监控')
    parser.add_argument('--mode', choices=['full', 'quick', 'logs', 'process', 'gpu', 'results'],
                       default='quick', help='监控模式')
    parser.add_argument('--log-pattern', default='log_0909/TF_DWT_*.log',
                       help='日志文件模式')
    
    args = parser.parse_args()
    
    monitor = TuningMonitor()
    
    if args.mode == 'full':
        monitor.start_monitoring(args.log_pattern)
    elif args.mode == 'quick':
        monitor.quick_status()
    elif args.mode == 'logs':
        monitor.monitor_logs(args.log_pattern)
    elif args.mode == 'process':
        monitor.monitor_processes()
    elif args.mode == 'gpu':
        monitor.monitor_gpu()
    elif args.mode == 'results':
        monitor.monitor_results()

if __name__ == "__main__":
    main()
