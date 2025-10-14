"""
Configuration file to reproduce TF-DWT experiment results from 20250924_204648
This configuration replicates the exact settings used to generate tfdwt_summary_stats_20250924_204648.csv
"""

import os

#######################
# Path Configuration
#######################

# Define paths for the datasets
P3_DATA_DIR = '../P3_Raw_Data_BIDS-Compatible'
AVO_DATA_DIR = '../ds005863'
LOG_DIR = './log_0909'

#######################
# Dataset Configuration
#######################

# Combined datasets configuration (as used in original experiment)
use_combined_datasets = True
data_dir = P3_DATA_DIR
dataset = 'use_combined_datasets'

# Nested Cross-Validation trial configuration (exact settings from log)
NESTED_CV_TRIALS_PER_SUBJECT_P3 = 20    # As logged: "trials_per_subject_P3: 20"
NESTED_CV_TRIALS_PER_SUBJECT_AVO = 200  # As logged: "trials_per_subject_AVO: 200"

#######################
# Experiment Configuration
#######################

# Electrode Configuration (exact setting from log)
electrode_list = 'all'  # As logged: "electrode_list: all"

# Model Configuration
classifier = 'EEGConformer'

# Training Configuration
separate_subject_classification = False
use_subject_layer = False

#######################
# Preprocessing Configuration
#######################

# Filtering
LOW_FREQ = 0.5
HIGH_FREQ = 30
RESAMPLE_FREQ = 128

# Trial window (in samples, relative to event)
TRIAL_START_OFFSET_SAMPLES = int(-0.1 * 128)  # -100ms before event
TRIAL_STOP_OFFSET_SAMPLES = int(1.0 * 128)     # 1 second after event

#######################
# Training Configuration (exact settings from log)
#######################

# Model and training hyperparameters (as logged)
BATCH_SIZE = 32          # As logged: "batch_size: 32"
MAX_EPOCHS = 500         # As logged: "max_epochs: 500"

# Dataset split ratios (as logged)
TRAIN_SIZE = 0.7         # As logged: "train/val/test: (0.7, 0.1, 0.2)"
VAL_SIZE = 0.1
TEST_SIZE = 0.2

#######################
# Subject and Trial Configuration
#######################

# Maximum number of subjects to use from each dataset
MAX_SUBJECTS_P3 = 40      # Maximum subjects from P3 dataset
MAX_SUBJECTS_AVO = 40     # Maximum subjects from AVO dataset

# Trial limits per subject (set to None for no limit)
MAX_TRIALS_PER_SUBJECT_TRAIN = None
MAX_TRIALS_PER_SUBJECT_VAL = None
MAX_TRIALS_PER_SUBJECT_TEST = None

# Traditional train/val/test split configuration (not used in TF-DWT)
FIXED_TRIALS_PER_SUBJECT_TRAIN = 60
FIXED_TRIALS_PER_SUBJECT_VAL = 20
FIXED_TRIALS_PER_SUBJECT_TEST = 10

# Random seeds for reproducibility (same as original)
seeds = [42, 123, 456, 789, 321]

#######################
# Nested Cross-Validation Configuration (exact settings)
#######################

# Enable nested cross-validation
USE_NESTED_CV = True

# Nested CV configuration (default values from main_tfdwt.py)
NESTED_CV_OUTER_FOLDS = 5      # 5-fold CV as standard
NESTED_CV_REPEATS = 5          # 5 repeats as standard
NESTED_CV_CONFIDENCE_LEVEL = 0.95  # 95% confidence intervals

#######################
# Model Configuration Details (exact settings from log)
#######################

# Input/Output dimensions
INPUT_WINDOW_SAMPLES = TRIAL_STOP_OFFSET_SAMPLES - TRIAL_START_OFFSET_SAMPLES
N_CLASSES = 2

# Training hyperparameters (as logged)
LEARNING_RATE = 0.01     # As logged: "learning_rate: 0.01"
WEIGHT_DECAY = 1e-4      # As logged: "weight_decay: 0.0001"
GAMMA = 0.7              # Learning rate decay factor
EARLY_STOPPING_PATIENCE = 50  # As logged: early stopping patience = 50
DROPOUT_RATE = 0.25      # As logged: "dropout_rate: 0.25"

# Data augmentation (exact settings from log)
USE_DATA_AUGMENTATION = True    # As logged: "use_data_augmentation: True"
NOISE_STD = 0.005              # As logged: "noise_std: 0.005"
TIME_SHIFT_RANGE = 5           # As logged: "time_shift_range: 5"
LABEL_SMOOTHING = 0.05         # As logged: "label_smoothing: 0.05"

# Small Dataset Overfitting Prevention Configuration
SMALL_DATASET_THRESHOLD = 1000
ENABLE_SMALL_DATASET_PROTECTIONS = False

# Small dataset specific settings
SMALL_DATASET_DROPOUT_RATE = 0.2
SMALL_DATASET_LEARNING_RATE = 0.01
SMALL_DATASET_WEIGHT_DECAY = 1e-4
SMALL_DATASET_EARLY_STOPPING_PATIENCE = 10
SMALL_DATASET_MAX_EPOCHS = 300
SMALL_DATASET_BATCH_SIZE = 16

# Enhanced preprocessing options (exact settings from log)
USE_ENHANCED_PREPROCESSING = True    # As logged: "use_enhanced_preprocessing: True"
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

# Performance tuning for SepConv1D models
SEPCONV1D_USE_WARMUP = True
SEPCONV1D_WARMUP_EPOCHS = 10
SEPCONV1D_WARMUP_FACTOR = 0.1

#######################
# Device Configuration (exact setting from log)
#######################

DEVICE_MODE = 'cuda'    # As logged: "device_mode: cuda"

#######################
# Performance Optimization Configuration
#######################

# Data caching options
USE_DATA_CACHE = True
CACHE_DIR = './cache'

# Parallel processing options
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = None
USE_THREADS = False

# Memory optimization options
ENABLE_MEMORY_OPTIMIZATION = True
MAX_MEMORY_MB = 2000
CHUNK_SIZE = 50
OPTIMIZE_DTYPES = True

# Verbose output
VERBOSE_PROCESSING = True

#######################
# Multi-Dataset Fusion Configuration (exact settings from log)
#######################

# 电极分布融合方法选择 (as logged)
ELECTRODE_FUSION_METHOD = 'none'    # As logged: "fusion_method: none"

# 图神经网络配置
GCN_HIDDEN_DIM = 64
GCN_NUM_LAYERS = 2
GCN_EMBEDDING_DIM = 128
GCN_DROPOUT = 0.3
GCN_LEARNING_RATE = 0.001

# 图增强配置
GRAPH_ENHANCEMENT_STRENGTH = 0.1
GRAPH_ADJACENCY_METHOD = 'knn'
GRAPH_K_NEIGHBORS = 3

#######################
# Domain Adaptation Configuration (exact settings from log)
#######################

# 领域自适应方法选择 (as logged)
DOMAIN_ADAPTATION_METHOD = 'none'    # As logged: "domain_adaptation: none"

# MS-MDA配置
MS_MDA_ADAPTATION_WEIGHT = 0.1
MS_MDA_ENSEMBLE_METHOD = 'weighted_average'
MS_MDA_HIDDEN_DIM = 256
MS_MDA_TEMPERATURE = 1.0

# 对抗性领域自适应配置
ADVERSARIAL_WEIGHT = 0.1
DISCRIMINATOR_HIDDEN_DIM = 128
DISCRIMINATOR_LEARNING_RATE = 0.0001
GRADIENT_REVERSAL_LAMBDA = 1.0

#######################
# Evaluation Configuration
#######################

# 评估模式配置
ENABLE_COMPREHENSIVE_EVALUATION = True
ENABLE_DOMAIN_ANALYSIS = True
ENABLE_SMALL_SAMPLE_ANALYSIS = False

# 小样本实验配置
SMALL_SAMPLE_SIZES = [5, 10, 15, 20]
SMALL_SAMPLE_SUBJECTS = [5, 10, 15]

# 随机种子配置
RANDOM_SEED = 42

#######################
# REPRODUCTION NOTES
#######################

"""
This configuration reproduces the exact experiment from 20250924_204648 with results:
- Overall accuracy: 0.6389 ± 0.0180
- 95% Confidence Interval: [0.6315, 0.6463]
- P3 Dataset - Mean Accuracy: 0.5884 | 95% CI: [0.5743, 0.6024]
- AVO Dataset - Mean Accuracy: 0.6523 | 95% CI: [0.6432, 0.6614]

Key settings extracted from log_0909/TF_DWT_results_20250924_202832.log:
- electrode_list: all
- fusion_method: none
- domain_adaptation: none
- use_enhanced_preprocessing: True
- batch_size: 32
- max_epochs: 500
- learning_rate: 0.01
- weight_decay: 0.0001
- dropout_rate: 0.25
- use_data_augmentation: True
- noise_std: 0.005
- time_shift_range: 5
- label_smoothing: 0.05
- trials_per_subject_P3: 20
- trials_per_subject_AVO: 200
- train/val/test: (0.7, 0.1, 0.2)
- device_mode: cuda

To use this configuration:
1. Backup current config.py: cp config.py config_backup.py
2. Replace with this file: cp config_reproduce_tfdwt_20250924_204648.py config.py
3. Run: python main_tfdwt.py
4. Restore original config: cp config_backup.py config.py
"""