#!/usr/bin/env python3
"""
GPU利用率监控脚本 - 实时监控GPU使用情况
"""

import subprocess
import time
import threading
import queue
import os
import sys

class GPUMonitor:
    def __init__(self):
        self.monitoring = False
        self.gpu_data = queue.Queue()
        
    def get_gpu_info(self):
        """获取GPU信息"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info = []
                for line in lines:
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 7:
                            gpu_info.append({
                                'index': parts[0],
                                'name': parts[1],
                                'utilization': int(parts[2]),
                                'memory_used': int(parts[3]),
                                'memory_total': int(parts[4]),
                                'temperature': int(parts[5]),
                                'power': float(parts[6]) if parts[6] != 'N/A' else 0
                            })
                return gpu_info
        except Exception as e:
            print(f"获取GPU信息失败: {e}")
        return []
    
    def monitor_worker(self, interval=1):
        """监控工作线程"""
        while self.monitoring:
            gpu_info = self.get_gpu_info()
            if gpu_info:
                self.gpu_data.put((time.time(), gpu_info))
            time.sleep(interval)
    
    def start_monitoring(self, interval=1):
        """开始监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_worker, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"📊 GPU监控已启动 (间隔: {interval}秒)")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        print("⏹️ GPU监控已停止")
    
    def print_current_status(self):
        """打印当前状态"""
        gpu_info = self.get_gpu_info()
        if not gpu_info:
            print("❌ 无法获取GPU信息")
            return
        
        print(f"\n🖥️ GPU状态 ({time.strftime('%H:%M:%S')})")
        print("="*80)
        
        for gpu in gpu_info:
            memory_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
            print(f"GPU {gpu['index']}: {gpu['name']}")
            print(f"  使用率: {gpu['utilization']:3d}% | 内存: {gpu['memory_used']:5d}/{gpu['memory_total']:5d}MB ({memory_percent:5.1f}%)")
            print(f"  温度: {gpu['temperature']:3d}°C | 功耗: {gpu['power']:6.1f}W")
            
            # 使用率状态指示
            if gpu['utilization'] > 80:
                status = "🔥 高负载"
            elif gpu['utilization'] > 50:
                status = "⚡ 中等负载"
            elif gpu['utilization'] > 20:
                status = "🟡 低负载"
            else:
                status = "😴 空闲"
            
            print(f"  状态: {status}")
            print()
    
    def monitor_continuous(self, duration=60):
        """连续监控指定时间"""
        print(f"🚀 开始连续监控 {duration} 秒")
        print("按 Ctrl+C 停止监控")
        print("="*80)
        
        self.start_monitoring(interval=2)
        
        try:
            start_time = time.time()
            while time.time() - start_time < duration:
                self.print_current_status()
                time.sleep(5)  # 每5秒更新一次显示
        except KeyboardInterrupt:
            print("\n⏹️ 监控被用户中断")
        finally:
            self.stop_monitoring()
    
    def analyze_gpu_usage(self):
        """分析GPU使用情况"""
        print("📈 GPU使用情况分析")
        print("="*50)
        
        gpu_info = self.get_gpu_info()
        if not gpu_info:
            print("❌ 无法获取GPU信息")
            return
        
        for gpu in gpu_info:
            memory_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
            
            print(f"GPU {gpu['index']}: {gpu['name']}")
            print(f"  当前使用率: {gpu['utilization']}%")
            print(f"  内存使用: {memory_percent:.1f}%")
            print(f"  温度: {gpu['temperature']}°C")
            print(f"  功耗: {gpu['power']:.1f}W")
            
            # 建议
            if gpu['utilization'] < 20:
                print("  💡 建议: GPU使用率较低，可以增加batch_size或并行度")
            elif gpu['utilization'] > 90:
                print("  ⚠️ 警告: GPU使用率很高，注意散热")
            
            if memory_percent < 30:
                print("  💡 建议: GPU内存使用率较低，可以增加模型复杂度或batch_size")
            elif memory_percent > 90:
                print("  ⚠️ 警告: GPU内存使用率很高，可能需要减少batch_size")
            
            print()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU监控工具')
    parser.add_argument('--mode', choices=['status', 'monitor', 'analyze'], 
                       default='status', help='监控模式')
    parser.add_argument('--duration', type=int, default=60, help='监控持续时间（秒）')
    
    args = parser.parse_args()
    
    monitor = GPUMonitor()
    
    if args.mode == 'status':
        monitor.print_current_status()
    elif args.mode == 'monitor':
        monitor.monitor_continuous(args.duration)
    elif args.mode == 'analyze':
        monitor.analyze_gpu_usage()

if __name__ == "__main__":
    main()
