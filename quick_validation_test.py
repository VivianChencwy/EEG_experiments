#!/usr/bin/env python3
"""
快速验证测试 - 直接修改config.py进行测试
"""

import os
import sys
import subprocess
import time
import json
import shutil
from pathlib import Path

def backup_and_modify_config(params):
    """备份原始config.py并创建修改版本"""
    # 备份原始config
    shutil.copy("config.py", "config_backup.py")
    
    # 读取原始config
    with open("config.py", "r") as f:
        content = f.read()
    
    # 修改关键参数
    modifications = {
        'MAX_EPOCHS = 500': f'MAX_EPOCHS = {params.get("MAX_EPOCHS", 5)}',
        'EARLY_STOPPING_PATIENCE = 50': f'EARLY_STOPPING_PATIENCE = {params.get("EARLY_STOPPING_PATIENCE", 3)}',
        'LEARNING_RATE = 0.01': f'LEARNING_RATE = {params.get("LEARNING_RATE", 0.01)}',
        'BATCH_SIZE = 32': f'BATCH_SIZE = {params.get("BATCH_SIZE", 32)}',
        'DROPOUT_RATE = 0.25': f'DROPOUT_RATE = {params.get("DROPOUT_RATE", 0.25)}',
        'classifier = \'EEGConformer\'': f'classifier = \'{params.get("classifier", "EEGConformer")}\'',
        'NESTED_CV_OUTER_FOLDS = 5': 'NESTED_CV_OUTER_FOLDS = 2',
        'NESTED_CV_REPEATS = 5': 'NESTED_CV_REPEATS = 1',
    }
    
    for old, new in modifications.items():
        content = content.replace(old, new)
    
    # 写入修改后的config
    with open("config.py", "w") as f:
        f.write(content)
    
    return "config_backup.py"

def restore_config(backup_path):
    """恢复原始config.py"""
    shutil.copy(backup_path, "config.py")
    os.unlink(backup_path)

def run_validation_test():
    """运行验证测试"""
    print("QUICK VALIDATION TEST")
    print("="*50)
    
    # 测试参数
    test_params = {
        'MAX_EPOCHS': 3,
        'EARLY_STOPPING_PATIENCE': 2,
        'LEARNING_RATE': 0.01,
        'BATCH_SIZE': 32,
        'DROPOUT_RATE': 0.25,
        'classifier': 'EEGConformer'
    }
    
    print(f"Test parameters: {test_params}")
    
    # 备份并修改config
    backup_path = backup_and_modify_config(test_params)
    
    try:
        print("\nStarting validation test...")
        start_time = time.time()
        
        # 运行main_tfdwt.py
        result = subprocess.run(
            [sys.executable, "main_tfdwt.py"],
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )
        
        duration = time.time() - start_time
        
        print(f"\nTest completed in {duration/60:.1f} minutes")
        
        if result.returncode != 0:
            print(f"Test FAILED with return code: {result.returncode}")
            print(f"Error: {result.stderr[:300]}...")
            return False, result.stderr
        
        # 提取准确率
        accuracy = extract_accuracy(result.stdout)
        print(f"Test SUCCESS - Accuracy: {accuracy:.4f}")
        
        return True, accuracy
        
    except subprocess.TimeoutExpired:
        print("Test TIMED OUT after 10 minutes")
        return False, "timeout"
    except Exception as e:
        print(f"Test FAILED with exception: {e}")
        return False, str(e)
    finally:
        # 恢复原始config
        restore_config(backup_path)
        print("Original config.py restored")

def extract_accuracy(output):
    """从输出中提取准确率"""
    import re
    
    patterns = [
        r'Overall accuracy:\s+([0-9.]+)',
        r'mean_accuracy[\'\"]*\s*[:\s=]+\s*([0-9.]+)',
        r'Mean Accuracy:\s+([0-9.]+)',
        r'Final Results: Overall Accuracy = ([0-9.]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            accuracies = [float(match) for match in matches if float(match) <= 1.0]
            if accuracies:
                return max(accuracies)
    
    return 0.0

if __name__ == "__main__":
    success, result = run_validation_test()
    
    if success:
        print(f"\n{'='*50}")
        print("VALIDATION TEST PASSED")
        print(f"Accuracy: {result:.4f}")
        print("Code logic is working correctly!")
        print("You can now increase epochs for full tuning.")
    else:
        print(f"\n{'='*50}")
        print("VALIDATION TEST FAILED")
        print(f"Error: {result}")
        print("Need to fix issues before proceeding.")
