#!/usr/bin/env python3
"""
完整调参运行脚本 - 一键运行所有调参策略
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, description, timeout=None):
    """运行命令并显示进度"""
    print(f"\n🚀 {description}")
    print(f"命令: {cmd}")
    print("="*80)
    
    try:
        if timeout:
            result = subprocess.run(cmd, shell=True, timeout=timeout)
        else:
            result = subprocess.run(cmd, shell=True)
        
        if result.returncode == 0:
            print(f"✅ {description} 完成")
            return True
        else:
            print(f"❌ {description} 失败 (返回码: {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} 超时")
        return False
    except KeyboardInterrupt:
        print(f"⏹️  {description} 被用户中断")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='完整调参运行脚本')
    parser.add_argument('--strategy', choices=['quick', 'layered', 'parallel', 'incremental', 'all'],
                       default='layered', help='调参策略')
    parser.add_argument('--trials', type=int, help='试验数量')
    parser.add_argument('--workers', type=int, help='工作进程数')
    parser.add_argument('--monitor', action='store_true', help='启动监控')
    parser.add_argument('--skip-validation', action='store_true', help='跳过验证步骤')
    
    args = parser.parse_args()
    
    print("🎯 TFDWT 完整调参系统")
    print("="*100)
    
    # 检查系统状态
    print("🔍 检查系统状态...")
    if not run_command("python complete_tuning_guide.py --check", "系统状态检查", timeout=30):
        print("❌ 系统检查失败，请检查环境配置")
        return
    
    # 快速验证（除非跳过）
    if not args.skip_validation:
        print("\n" + "="*100)
        print("1️⃣ 快速验证阶段")
        print("="*100)
        
        if not run_command("python quick_validation_test.py", "快速验证", timeout=600):
            print("❌ 快速验证失败，请检查代码")
            return
    
    # 根据策略运行调参
    print("\n" + "="*100)
    print("2️⃣ 主要调参阶段")
    print("="*100)
    
    success = False
    
    if args.strategy == 'quick':
        success = run_command("python quick_validation_test.py", "快速验证")
    
    elif args.strategy == 'layered':
        success = run_command("python layered_tuning.py", "分层调参")
    
    elif args.strategy == 'parallel':
        cmd = "python parallel_tuning.py"
        if args.trials:
            cmd += f" --trials {args.trials}"
        if args.workers:
            cmd += f" --workers {args.workers}"
        success = run_command(cmd, "并行调参")
    
    elif args.strategy == 'incremental':
        cmd = "python incremental_tuning.py"
        if args.trials:
            cmd += f" --trials {args.trials}"
        success = run_command(cmd, "增量调参")
    
    elif args.strategy == 'all':
        # 运行所有策略
        strategies = [
            ("python layered_tuning.py", "分层调参"),
            ("python parallel_tuning.py --trials 10 --workers 2", "并行调参"),
            ("python incremental_tuning.py --trials 5", "增量调参")
        ]
        
        for cmd, desc in strategies:
            if run_command(cmd, desc):
                print(f"✅ {desc} 完成，继续下一个...")
            else:
                print(f"❌ {desc} 失败，跳过...")
                continue
    
    # 启动监控（如果请求）
    if args.monitor and success:
        print("\n" + "="*100)
        print("3️⃣ 监控阶段")
        print("="*100)
        
        print("📊 启动实时监控...")
        print("按 Ctrl+C 停止监控")
        
        try:
            subprocess.run("python monitor_tuning.py --mode full", shell=True)
        except KeyboardInterrupt:
            print("\n⏹️  监控已停止")
    
    # 最终报告
    print("\n" + "="*100)
    print("📊 调参完成报告")
    print("="*100)
    
    # 检查结果文件
    result_dirs = [
        "layered_tuning_results",
        "parallel_tuning_results",
        "incremental_tuning_results",
        "quick_test_results"
    ]
    
    print("📁 结果文件:")
    for dir_name in result_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            files = list(dir_path.glob("*"))
            print(f"   ✅ {dir_name}/ ({len(files)} 个文件)")
        else:
            print(f"   ❌ {dir_name}/ - 不存在")
    
    # 显示最新日志
    log_files = list(Path(".").glob("log_0909/TF_DWT_*.log"))
    if log_files:
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        print(f"\n📄 最新日志: {latest_log}")
        
        # 显示最后几行
        try:
            result = subprocess.run(['tail', '-3', str(latest_log)], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("最新结果:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        print(f"   {line.strip()}")
        except:
            pass
    
    print(f"\n🎉 调参任务完成！")
    print("="*100)

if __name__ == "__main__":
    main()
