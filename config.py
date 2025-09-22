"""
Configuration file for AVO (Active Visual Oddball) EEG experiments
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

# Option 1: P3 dataset only 
# data_dir = P3_DATA_DIR
# dataset = 'P3 Raw Data BIDS-Compatible'
# use_combined_datasets = False

# Option 2: ds005863 only
# data_dir = AVO_DATA_DIR
# dataset = 'ds005863'
# use_combined_datasets = False

# Option 3: Both datasets combined
use_combined_datasets = True
data_dir = P3_DATA_DIR
dataset = 'use_combined_datasets'


# Nested Cross-Validation trial configuration (only used when USE_NESTED_CV = True)
# Specify how many trials to use per subject for each dataset in nested CV
# The 5-fold CV will automatically handle train/val/test splits, so you only need to specify total trials
NESTED_CV_TRIALS_PER_SUBJECT_P3 = 200    # Number of trials per P3 subject for nested CV
NESTED_CV_TRIALS_PER_SUBJECT_AVO = 20   # Number of trials per AVO subject for nested CV


#######################
# Experiment Configuration
#######################

# Electrode Configuration
#electrode_list = 'common'
electrode_list = 'all'

# Model Configuration
# Available options:
# - 'lda': Linear Discriminant Analysis
# - 'ShallowFBCSPNet': Standard shallow CNN for EEG (original baseline)
# - 'EEGNetv4': Compact CNN specifically designed for EEG 
# - 'Deep4Net': Deep convolutional network for EEG
# - 'EEGConformer': CNN-Transformer hybrid for EEG (state-of-the-art)
# - 'EEGChannelNet': CNN with channel-wise attention mechanism
# - 'SepConv1D': Enhanced separable convolution model with multi-scale features
# - 'SepConv1DLite': Ultra-lightweight version with residual connections (recommended for small datasets)

#classifier = 'EEGNet'
classifier = 'EEGConformer'
#classifier = 'ShallowFBCSPNet'
#classifier = 'DeepConvNet' #problem
#classifier = 'EEGChannelNet'
#classifier = 'SepConv1D'
#classifier = 'SepConv1DLite'  # Try the lite version first for better accuracy
#classifier = 'lda'

# Training Configuration
separate_subject_classification = False

# Subject Layer Configuration (only applies to ShallowFBCSPNet + pooled training)
use_subject_layer = False

#######################
# Preprocessing Configuration
#######################

# Filtering
LOW_FREQ = 0.5
HIGH_FREQ = 30
RESAMPLE_FREQ = 128

# Trial window (in samples, relative to event)
# Window: ~150ms before event to 1s after event (total ~1.15s)
# This captures the baseline period before stimulus and the full ERP response
TRIAL_START_OFFSET_SAMPLES = int(-0.1 * 128)  # -100ms before event (-19 samples, baseline period)
TRIAL_STOP_OFFSET_SAMPLES = int(1.0 * 128)     # 1 second after event (128 samples)

# Notes:
# - Pre-stimulus period (-150ms to 0ms) serves as baseline for baseline correction
# - Post-stimulus period (0ms to 1000ms) captures the full ERP components including P3
# - Total window length: 147 samples (1148ms) at 128Hz
# - Baseline correction (if enabled) will use the pre-stimulus period

#######################
# Training Configuration
#######################

# Model and training hyperparameters
BATCH_SIZE = 32
MAX_EPOCHS = 500

# Dataset split ratios (must sum to ≤ 1.0)
TRAIN_SIZE = 0.7
VAL_SIZE = 0.1
TEST_SIZE = 0.2

#######################
# Subject and Trial Configuration
#######################

# Maximum number of subjects to use from each dataset
MAX_SUBJECTS_P3 = 40      # Maximum subjects from P3 dataset
MAX_SUBJECTS_AVO = 40     # Maximum subjects from AVO dataset

# Trial limits per subject (set to None for no limit)
# These control how many trials to use from each subject's data
MAX_TRIALS_PER_SUBJECT_TRAIN = None    # None = use all available trials
MAX_TRIALS_PER_SUBJECT_VAL = None      # None = use all available trials  
MAX_TRIALS_PER_SUBJECT_TEST = None     # None = use all available trials

# Traditional train/val/test split configuration (only used when USE_NESTED_CV = False)
# Set these to specific numbers if you want exact trial counts
FIXED_TRIALS_PER_SUBJECT_TRAIN = 60  # e.g., 100 for exactly 100 train trials per subject
FIXED_TRIALS_PER_SUBJECT_VAL = 20    # e.g., 20 for exactly 20 val trials per subject
FIXED_TRIALS_PER_SUBJECT_TEST = 10   # e.g., 30 for exactly 30 test trials per subject



# Examples for different experimental designs:
#
# Balanced design (same trials per subject for both datasets):
# NESTED_CV_TRIALS_PER_SUBJECT_P3 = 50
# NESTED_CV_TRIALS_PER_SUBJECT_AVO = 50
#
# Unbalanced design (different trials per subject - system handles stratification):
# NESTED_CV_TRIALS_PER_SUBJECT_P3 = 60    # P3 has more trials available
# NESTED_CV_TRIALS_PER_SUBJECT_AVO = 30   # AVO has fewer trials available
#
# Quick testing (small number of trials):
# NESTED_CV_TRIALS_PER_SUBJECT_P3 = 20
# NESTED_CV_TRIALS_PER_SUBJECT_AVO = 20

# Example configurations:
# 
# To use exactly 50 subjects from each dataset with 100/20/30 trials per subject:
# MAX_SUBJECTS_P3 = 50
# MAX_SUBJECTS_AVO = 50
# FIXED_TRIALS_PER_SUBJECT_TRAIN = 100
# FIXED_TRIALS_PER_SUBJECT_VAL = 20
# FIXED_TRIALS_PER_SUBJECT_TEST = 30
#
# To limit to maximum 200 train trials per subject (but use all val/test):
# MAX_TRIALS_PER_SUBJECT_TRAIN = 200
# MAX_TRIALS_PER_SUBJECT_VAL = None
# MAX_TRIALS_PER_SUBJECT_TEST = None

# Random seeds for reproducibility
seeds = [42]#, 123, 456, 789, 321]

#######################
# Nested Cross-Validation Configuration
#######################

# Enable nested cross-validation (politically correct approach)
USE_NESTED_CV = True

# Nested CV configuration
NESTED_CV_OUTER_FOLDS = 5      # Outer CV folds for performance estimation
# NESTED_CV_INNER_FOLDS = 3      # 已删除：不再使用内层超参数调优
NESTED_CV_REPEATS = 1         # Number of times to repeat the entire process
NESTED_CV_CONFIDENCE_LEVEL = 0.95  # Confidence level for intervals

#######################
# Model Configuration Details
#######################

# Input/Output dimensions
INPUT_WINDOW_SAMPLES = TRIAL_STOP_OFFSET_SAMPLES - TRIAL_START_OFFSET_SAMPLES  # Total window samples
N_CLASSES = 2

# Training hyperparameters
LEARNING_RATE = 0.01   # Increased for SepConv1D models to improve learning
WEIGHT_DECAY = 1e-4     # Moderate regularization
GAMMA = 0.7  # Learning rate decay factor (less aggressive)
EARLY_STOPPING_PATIENCE = 50 # Balanced patience
DROPOUT_RATE = 0.25     # Reduced dropout for lightweight models

# Data augmentation (enabled for better generalization)
USE_DATA_AUGMENTATION = True
NOISE_STD = 0.005       # Reduced noise
TIME_SHIFT_RANGE = 5    # Small time shifts (in samples)
LABEL_SMOOTHING = 0.05  # Reduced label smoothing

# Small Dataset Overfitting Prevention Configuration
# These settings are automatically applied when using SepConv1D or when detected small sample size
SMALL_DATASET_THRESHOLD = 1000          # Threshold to consider dataset as "small" (total samples across all subjects)
ENABLE_SMALL_DATASET_PROTECTIONS = False # Enable automatic overfitting prevention for small datasets

# Small dataset specific settings (applied automatically when conditions are met)
SMALL_DATASET_DROPOUT_RATE = 0.2       # Lower dropout for small datasets (vs 0.3 default)
SMALL_DATASET_LEARNING_RATE = 0.01    # Higher initial learning rate
SMALL_DATASET_WEIGHT_DECAY = 1e-4      # Stronger L2 regularization  
SMALL_DATASET_EARLY_STOPPING_PATIENCE = 20  # Lower patience to avoid overfitting
SMALL_DATASET_MAX_EPOCHS = 300          # Fewer max epochs
SMALL_DATASET_BATCH_SIZE = 16           # Smaller batch size for better gradient estimates

# Enhanced preprocessing options
USE_ENHANCED_PREPROCESSING = True    # Enable advanced preprocessing features
REMOVE_ARTIFACTS = True              # Use ICA for artifact removal
BASELINE_CORRECT = True              # Apply baseline correction
EXTRACT_FREQUENCY_FEATURES = True    # Add frequency domain features
APPLY_NOTCH_FILTER = True            # Remove power line interference

# EEGConformer specific parameters
CONFORMER_CONV_SPATIAL_DIM = 40      # Spatial convolution output channels
CONFORMER_CONV_TEMPORAL_DIM = 25     # Temporal convolution output channels  
CONFORMER_EMBEDDING_DIM = 40         # Transformer embedding dimension
CONFORMER_NUM_HEADS = 10             # Number of attention heads
CONFORMER_NUM_LAYERS = 3             # Number of transformer layers
CONFORMER_ACTIVATION = 'gelu'        # Transformer activation function

# SepConv1D specific parameters (optimized for better accuracy while staying lightweight)
SEPCONV1D_FILTERS = 48               # Increased filters for better representation capacity
SEPCONV1D_KERNEL_SIZE = 16           # Temporal kernel size for separable convolution
SEPCONV1D_STRIDE = 8                 # Stride for downsampling (reduces parameters)
SEPCONV1D_PADDING = 4                # Padding for temporal convolution

# Performance tuning for SepConv1D models
SEPCONV1D_USE_WARMUP = True          # Use learning rate warmup for better convergence
SEPCONV1D_WARMUP_EPOCHS = 10         # Number of warmup epochs
SEPCONV1D_WARMUP_FACTOR = 0.1        # Starting learning rate factor during warmup

#######################
# Device Configuration
#######################

# 设备选择配置
# 'auto': 自动检测 (有GPU用GPU，无GPU用CPU) - 推荐设置
# 'cpu': 强制使用CPU (即使有GPU也不使用，适合调试或CPU-only环境)
# 'cuda': 强制使用CUDA (如果无GPU会报错)
#
# 使用说明：
# - 要强制CPU运行，改为 DEVICE_MODE = 'cpu'
# - 要强制GPU运行，改为 DEVICE_MODE = 'cuda'
# - 自动选择设备，保持 DEVICE_MODE = 'auto'
DEVICE_MODE = 'auto'

#######################
# Performance Optimization Configuration
#######################

# Data caching options
USE_DATA_CACHE = True
CACHE_DIR = './cache'

# Parallel processing options  
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = None  # None = use all available CPU cores
USE_THREADS = False  # Use ProcessPoolExecutor by default

# Memory optimization options
ENABLE_MEMORY_OPTIMIZATION = True
MAX_MEMORY_MB = 2000  # Maximum memory usage in MB
CHUNK_SIZE = 50  # Number of windows to process at once
OPTIMIZE_DTYPES = True  # Convert float64 to float32 where possible

# Verbose output
VERBOSE_PROCESSING = True

#######################
# Multi-Dataset Fusion Configuration
#######################

# 电极分布融合方法选择
# 'none': 不使用融合方法，保持baseline状态
# 'graph_gcn': 基于图神经网络的通用特征空间融合
# 'graph_enhanced': 轻量级GCN特征增强 + 主干模型
ELECTRODE_FUSION_METHOD = 'none'

# 图神经网络配置
GCN_HIDDEN_DIM = 64            # GCN隐藏层维度
GCN_NUM_LAYERS = 2             # GCN层数
GCN_EMBEDDING_DIM = 128        # 图嵌入向量维度
GCN_DROPOUT = 0.3              # GCN dropout率
GCN_LEARNING_RATE = 0.001      # GCN专用学习率

# 图增强配置
GRAPH_ENHANCEMENT_STRENGTH = 0.1    # 图增强的强度 (0.0-1.0)
GRAPH_ADJACENCY_METHOD = 'knn'      # 邻接矩阵构建方法: 'knn', 'threshold'
GRAPH_K_NEIGHBORS = 3               # KNN图的邻居数量

#######################
# Domain Adaptation Configuration
#######################

# 领域自适应方法选择
# 'none': 不使用领域自适应
# 'ms_mda': 多源边缘分布自适应
# 'adversarial': 对抗性领域自适应
DOMAIN_ADAPTATION_METHOD = 'none'

# MS-MDA配置
MS_MDA_ADAPTATION_WEIGHT = 0.1         # 适应损失权重
MS_MDA_ENSEMBLE_METHOD = 'weighted_average'  # 集成方法: 'average', 'weighted_average', 'voting'
MS_MDA_HIDDEN_DIM = 256                # 适配分支隐藏维度
MS_MDA_TEMPERATURE = 1.0               # softmax温度参数

# 对抗性领域自适应配置
ADVERSARIAL_WEIGHT = 0.1               # 对抗损失权重
DISCRIMINATOR_HIDDEN_DIM = 128         # 判别器隐藏维度
DISCRIMINATOR_LEARNING_RATE = 0.0001   # 判别器学习率
GRADIENT_REVERSAL_LAMBDA = 1.0         # 梯度反转强度

#######################
# Evaluation Configuration
#######################

# 评估模式配置
ENABLE_COMPREHENSIVE_EVALUATION = True  # 启用全面评估
ENABLE_DOMAIN_ANALYSIS = True          # 启用域间分析
ENABLE_SMALL_SAMPLE_ANALYSIS = False    # 启用小样本分析

# 小样本实验配置 (用于验证低资源效果)
SMALL_SAMPLE_SIZES = [5, 10, 15, 20]   # 每个subject使用的trial数量
SMALL_SAMPLE_SUBJECTS = [5, 10, 15]    # 每个数据集使用的subject数量

# 随机种子配置
RANDOM_SEED = 42