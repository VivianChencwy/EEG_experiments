#!/usr/bin/env python3
"""
简化的调参测试脚本 - 避免复杂的sys变量问题
"""

import os
import sys
import subprocess
import time
import json
import tempfile
import shutil
from pathlib import Path

def create_simple_config(params, trial_id):
    """创建简化的配置文件"""
    config_content = f'''#!/usr/bin/env python3
"""
Configuration file for AVO (Active Visual Oddball) EEG experiments
"""

import os

# Path Configuration
P3_DATA_DIR = '../P3_Raw_Data_BIDS-Compatible'
AVO_DATA_DIR = '../ds005863'
LOG_DIR = './log_0909'

# Dataset Configuration
use_combined_datasets = True
data_dir = P3_DATA_DIR
dataset = 'use_combined_datasets'

# Nested Cross-Validation trial configuration
NESTED_CV_TRIALS_PER_SUBJECT_P3 = 20    
NESTED_CV_TRIALS_PER_SUBJECT_AVO = 200  

# Experiment Configuration
electrode_list = 'all'

# Model Configuration
classifier = '{params.get("classifier", "EEGConformer")}'

# Training Configuration
separate_subject_classification = False
use_subject_layer = False

# Preprocessing Configuration
LOW_FREQ = 0.5
HIGH_FREQ = 30
RESAMPLE_FREQ = 128
TRIAL_START_OFFSET_SAMPLES = int(-0.1 * 128)
TRIAL_STOP_OFFSET_SAMPLES = int(1.0 * 128)

# Training Configuration
BATCH_SIZE = {params.get("BATCH_SIZE", 32)}
MAX_EPOCHS = {params.get("MAX_EPOCHS", 500)}

# Dataset split ratios
TRAIN_SIZE = 0.7
VAL_SIZE = 0.1
TEST_SIZE = 0.2

# Subject and Trial Configuration
MAX_SUBJECTS_P3 = 40
MAX_SUBJECTS_AVO = 40
MAX_TRIALS_PER_SUBJECT_TRAIN = None
MAX_TRIALS_PER_SUBJECT_VAL = None
MAX_TRIALS_PER_SUBJECT_TEST = None
FIXED_TRIALS_PER_SUBJECT_TRAIN = 60
FIXED_TRIALS_PER_SUBJECT_VAL = 20
FIXED_TRIALS_PER_SUBJECT_TEST = 10

# Random seeds
seeds = [42, 123, 456, 789, 321]

# Nested Cross-Validation Configuration
USE_NESTED_CV = True
NESTED_CV_OUTER_FOLDS = 2  # 减少折叠数用于快速测试
NESTED_CV_REPEATS = 1      # 减少重复次数用于快速测试
NESTED_CV_CONFIDENCE_LEVEL = 0.95

# Model Configuration Details
INPUT_WINDOW_SAMPLES = TRIAL_STOP_OFFSET_SAMPLES - TRIAL_START_OFFSET_SAMPLES
N_CLASSES = 2

# Training hyperparameters
LEARNING_RATE = {params.get("LEARNING_RATE", 0.01)}
WEIGHT_DECAY = {params.get("WEIGHT_DECAY", 1e-4)}
GAMMA = 0.7
EARLY_STOPPING_PATIENCE = {params.get("EARLY_STOPPING_PATIENCE", 50)}
DROPOUT_RATE = {params.get("DROPOUT_RATE", 0.25)}

# Data augmentation
USE_DATA_AUGMENTATION = True
NOISE_STD = {params.get("NOISE_STD", 0.005)}
TIME_SHIFT_RANGE = {params.get("TIME_SHIFT_RANGE", 5)}
LABEL_SMOOTHING = {params.get("LABEL_SMOOTHING", 0.05)}

# Small Dataset Overfitting Prevention Configuration
SMALL_DATASET_THRESHOLD = 1000
ENABLE_SMALL_DATASET_PROTECTIONS = False
SMALL_DATASET_DROPOUT_RATE = 0.2
SMALL_DATASET_LEARNING_RATE = 0.01
SMALL_DATASET_WEIGHT_DECAY = 1e-4
SMALL_DATASET_EARLY_STOPPING_PATIENCE = 20
SMALL_DATASET_MAX_EPOCHS = 300
SMALL_DATASET_BATCH_SIZE = 16

# Enhanced preprocessing options
USE_ENHANCED_PREPROCESSING = True
REMOVE_ARTIFACTS = True
BASELINE_CORRECT = True
EXTRACT_FREQUENCY_FEATURES = True
APPLY_NOTCH_FILTER = True

# EEGConformer specific parameters
CONFORMER_CONV_SPATIAL_DIM = 40
CONFORMER_CONV_TEMPORAL_DIM = 25
CONFORMER_EMBEDDING_DIM = 40
CONFORMER_NUM_HEADS = 10
CONFORMER_NUM_LAYERS = 3
CONFORMER_ACTIVATION = 'gelu'

# SepConv1D specific parameters
SEPCONV1D_FILTERS = 48
SEPCONV1D_KERNEL_SIZE = 16
SEPCONV1D_STRIDE = 8
SEPCONV1D_PADDING = 4
SEPCONV1D_USE_WARMUP = True
SEPCONV1D_WARMUP_EPOCHS = 10
SEPCONV1D_WARMUP_FACTOR = 0.1

# Device Configuration
DEVICE_MODE = 'auto'

# Performance Optimization Configuration
USE_DATA_CACHE = True
CACHE_DIR = './cache'
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = None
USE_THREADS = False
ENABLE_MEMORY_OPTIMIZATION = True
MAX_MEMORY_MB = 2000
CHUNK_SIZE = 50
OPTIMIZE_DTYPES = True
VERBOSE_PROCESSING = True

# Multi-Dataset Fusion Configuration
ELECTRODE_FUSION_METHOD = 'none'
GCN_HIDDEN_DIM = 64
GCN_NUM_LAYERS = 2
GCN_EMBEDDING_DIM = 128
GCN_DROPOUT = 0.3
GCN_LEARNING_RATE = 0.001
GRAPH_ENHANCEMENT_STRENGTH = 0.1
GRAPH_ADJACENCY_METHOD = 'knn'
GRAPH_K_NEIGHBORS = 3

# Domain Adaptation Configuration
DOMAIN_ADAPTATION_METHOD = 'none'
MS_MDA_ADAPTATION_WEIGHT = 0.1
MS_MDA_ENSEMBLE_METHOD = 'weighted_average'
MS_MDA_HIDDEN_DIM = 256
MS_MDA_TEMPERATURE = 1.0
ADVERSARIAL_WEIGHT = 0.1
DISCRIMINATOR_HIDDEN_DIM = 128
DISCRIMINATOR_LEARNING_RATE = 0.0001
GRADIENT_REVERSAL_LAMBDA = 1.0

# Evaluation Configuration
ENABLE_COMPREHENSIVE_EVALUATION = True
ENABLE_DOMAIN_ANALYSIS = True
ENABLE_SMALL_SAMPLE_ANALYSIS = False
SMALL_SAMPLE_SIZES = [5, 10, 15, 20]
SMALL_SAMPLE_SUBJECTS = [5, 10, 15]
RANDOM_SEED = 42
'''
    
    config_path = f"temp_config_trial_{trial_id}.py"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path

def run_single_trial(params, trial_id):
    """运行单个试验"""
    print(f"\n{'='*60}")
    print(f"TRIAL {trial_id + 1} STARTING")
    print(f"{'='*60}")
    print(f"Key Parameters:")
    for key in ['LEARNING_RATE', 'classifier', 'BATCH_SIZE', 'DROPOUT_RATE']:
        if key in params:
            print(f"   {key}: {params[key]}")
    
    start_time = time.time()
    
    try:
        # 创建临时配置文件
        config_path = create_simple_config(params, trial_id)
        
        # 设置环境变量
        env = os.environ.copy()
        env['CONFIG_OVERRIDE_PATH'] = os.path.abspath(config_path)
        
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
        
        # 清理临时文件
        if os.path.exists(config_path):
            os.unlink(config_path)
        
        if result.returncode != 0:
            print(f"TRIAL {trial_id + 1} FAILED after {duration/60:.1f} minutes")
            print(f"Error: {result.stderr[:200]}...")
            return -1.0, {'error': result.stderr, 'duration': duration}
        
        # 提取准确率
        accuracy = extract_accuracy(result.stdout)
        
        print(f"TRIAL {trial_id + 1} COMPLETED")
        print(f"Duration: {duration/60:.1f} minutes")
        print(f"Accuracy: {accuracy:.4f}")
        print("="*60)
        
        return accuracy, {'duration': duration, 'stdout': result.stdout[:500]}
        
    except subprocess.TimeoutExpired:
        print(f"TRIAL {trial_id + 1} TIMED OUT")
        return -1.0, {'error': 'timeout'}
    except Exception as e:
        print(f"TRIAL {trial_id + 1} FAILED: {e}")
        return -1.0, {'error': str(e)}

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

def main():
    """主函数"""
    print("SIMPLE TUNING TEST - 3 trials")
    print("="*60)
    
    # 定义测试参数
    test_params = [
        {
            'LEARNING_RATE': 0.01,
            'BATCH_SIZE': 32,
            'DROPOUT_RATE': 0.25,
            'classifier': 'EEGConformer',
            'MAX_EPOCHS': 5,  # 极低epochs用于快速验证
            'EARLY_STOPPING_PATIENCE': 3,
            'NOISE_STD': 0.005,
            'TIME_SHIFT_RANGE': 5,
            'LABEL_SMOOTHING': 0.05,
            'WEIGHT_DECAY': 1e-4
        },
        {
            'LEARNING_RATE': 0.005,
            'BATCH_SIZE': 24,
            'DROPOUT_RATE': 0.3,
            'classifier': 'EEGNetv4',
            'MAX_EPOCHS': 5,
            'EARLY_STOPPING_PATIENCE': 3,
            'NOISE_STD': 0.01,
            'TIME_SHIFT_RANGE': 3,
            'LABEL_SMOOTHING': 0.1,
            'WEIGHT_DECAY': 1e-3
        },
        {
            'LEARNING_RATE': 0.02,
            'BATCH_SIZE': 48,
            'DROPOUT_RATE': 0.2,
            'classifier': 'SepConv1DLite',
            'MAX_EPOCHS': 5,
            'EARLY_STOPPING_PATIENCE': 3,
            'NOISE_STD': 0.002,
            'TIME_SHIFT_RANGE': 8,
            'LABEL_SMOOTHING': 0.02,
            'WEIGHT_DECAY': 1e-5
        }
    ]
    
    results = []
    best_score = -1.0
    best_params = None
    
    for i, params in enumerate(test_params):
        score, metadata = run_single_trial(params, i)
        results.append({
            'trial_id': i,
            'params': params,
            'score': score,
            'metadata': metadata
        })
        
        if score > best_score:
            best_score = score
            best_params = params
            print(f"\nNEW BEST SCORE: {score:.4f}")
    
    # 保存结果
    os.makedirs("simple_test_results", exist_ok=True)
    with open("simple_test_results/results.json", "w") as f:
        json.dump({
            'best_score': best_score,
            'best_params': best_params,
            'all_results': results
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SIMPLE TUNING TEST COMPLETED")
    print(f"{'='*60}")
    print(f"Best accuracy: {best_score:.4f}")
    print(f"Best parameters: {best_params}")
    print(f"Results saved to: simple_test_results/results.json")

if __name__ == "__main__":
    main()
