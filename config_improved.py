"""
Improved configuration file with advanced preprocessing and model options
"""

import os

#######################
# Path Configuration
#######################

# Define paths for the datasets
P3_DATA_DIR = '../P3_Raw_Data_BIDS-Compatible'
AVO_DATA_DIR = '../ds005863'
LOG_DIR = './log_improved'

#######################
# Dataset Configuration
#######################

# Option 1: P3 dataset only 
# data_dir = P3_DATA_DIR
# dataset = 'P3 Raw Data BIDS-Compatible'
# use_combined_datasets = False

# Option 2: ds005863 only (current focus)
data_dir = AVO_DATA_DIR
dataset = 'ds005863'
use_combined_datasets = False

# Option 3: Both datasets combined
# use_combined_datasets = True
# data_dir = P3_DATA_DIR
# dataset = 'use_combined_datasets'

#######################
# Experiment Configuration
#######################

# Electrode Configuration
electrode_list = 'all'  # Use all available channels
# electrode_list = 'common'

# Advanced Model Configuration
# Options: 'ShallowFBCSPNet', 'attention_shallow', 'transformer', 'hybrid', 'lda'
classifier = 'attention_shallow'
# classifier = 'transformer'
# classifier = 'hybrid'
# classifier = 'ShallowFBCSPNet'

# Training Configuration
separate_subject_classification = False  # Use pooled training for better generalization

# Subject Layer Configuration (only applies to pooled training)
use_subject_layer = False

# Advanced Preprocessing Configuration
use_advanced_preprocessing = True
use_ica_artifact_removal = True
use_csp_spatial_filtering = True
n_csp_components = 8
ica_n_components = 15

#######################
# Advanced Preprocessing Configuration
#######################

# Filtering - Optimized for ERP/P300
LOW_FREQ = 0.5   # Standard high-pass
HIGH_FREQ = 30   # Standard low-pass
RESAMPLE_FREQ = 128

# ERP-optimized frequency band for advanced preprocessing
ERP_FREQ_BAND = (8, 30)  # Better for P300/ERP detection

# Trial window configuration
TRIAL_START_OFFSET_SAMPLES = 0
TRIAL_STOP_OFFSET_SAMPLES = int(1.0 * 128)  # 1 second at 128 Hz

# Alternative window sizes for optimization
WINDOW_SIZES = [
    int(0.5 * 128),  # 0.5 seconds
    int(1.0 * 128),  # 1.0 seconds  
    int(1.5 * 128),  # 1.5 seconds
]

#######################
# Advanced Training Configuration
#######################

# Model and training hyperparameters
BATCH_SIZE = 32
MAX_EPOCHS = 300  # Increased for better convergence

# Advanced learning rate configuration
LEARNING_RATE = 0.0005  # Lower initial LR for stability
WEIGHT_DECAY = 5e-4     # Slightly higher regularization
GAMMA = 0.5             # Learning rate decay factor

# Advanced scheduler configuration
USE_COSINE_ANNEALING = True
COSINE_T_MAX = 50
USE_WARMUP = True
WARMUP_EPOCHS = 10

# Dataset split ratios
TRAIN_SIZE = 0.7
VAL_SIZE = 0.1
TEST_SIZE = 0.2

# Random seeds for reproducibility
seeds = [42, 123, 456, 789, 321]  # Multiple seeds for robust evaluation

#######################
# Model Configuration Details
#######################

# Input/Output dimensions
INPUT_WINDOW_SAMPLES = int(1.0 * 128)  # Will be adjusted based on selected window size
N_CLASSES = 2

# Advanced regularization
EARLY_STOPPING_PATIENCE = 30  # Increased patience
DROPOUT_RATE = 0.3  # Optimized dropout

# Advanced loss configuration
USE_FOCAL_LOSS = True
FOCAL_ALPHA = 1.0
FOCAL_GAMMA = 2.0
USE_LABEL_SMOOTHING = True
LABEL_SMOOTHING = 0.1

#######################
# Advanced Data Augmentation Configuration
#######################

# Enable advanced augmentations
USE_ADVANCED_AUGMENTATION = True

# Augmentation parameters
AUGMENTATION_CONFIG = {
    'noise_std': 0.01,
    'time_shift_range': 10,
    'magnitude_warp_sigma': 0.15,
    'time_warp_sigma': 0.1,
    'channel_dropout_prob': 0.05,
    'temporal_cutout_size': 8,
    'mixup_alpha': 0.4,
    'use_mixup': True,
    'freq_shift_delta': 1.5,
    'amplitude_scale_range': (0.9, 1.1)
}

# Legacy augmentation (for compatibility)
USE_DATA_AUGMENTATION = USE_ADVANCED_AUGMENTATION
NOISE_STD = AUGMENTATION_CONFIG['noise_std']
TIME_SHIFT_RANGE = AUGMENTATION_CONFIG['time_shift_range']

#######################
# Hyperparameter Optimization Configuration
#######################

# Enable hyperparameter optimization
ENABLE_HYPERPARAMETER_OPTIMIZATION = False  # Set to True for full optimization

# Optimization search space
HYPERPARAMETER_SEARCH_SPACE = {
    'learning_rate': [0.0001, 0.0005, 0.001, 0.002],
    'batch_size': [16, 32, 64],
    'dropout_rate': [0.2, 0.3, 0.4, 0.5],
    'window_size_idx': [0, 1, 2],  # Index into WINDOW_SIZES
    'weight_decay': [1e-4, 5e-4, 1e-3],
    'ica_components': [10, 15, 20],
    'csp_components': [6, 8, 10, 12]
}

# Quick optimization (subset of parameters)
QUICK_OPTIMIZATION_SPACE = {
    'learning_rate': [0.0005, 0.001],
    'dropout_rate': [0.2, 0.3],
    'window_size_idx': [0, 1]
}

# Number of optimization trials
N_OPTIMIZATION_TRIALS = 50
N_QUICK_TRIALS = 10

#######################
# Model Architecture Specific Configuration
#######################

# Transformer model configuration
TRANSFORMER_CONFIG = {
    'd_model': 128,
    'n_heads': 8,
    'n_layers': 4,
    'd_ff': 512,
    'dropout': DROPOUT_RATE
}

# Hybrid CNN-Transformer configuration
HYBRID_CONFIG = {
    'd_model': 128,
    'n_heads': 8,
    'n_transformer_layers': 3,
    'dropout': DROPOUT_RATE
}

# Attention-enhanced ShallowFBCSPNet configuration
ATTENTION_SHALLOW_CONFIG = {
    'pool_mode': 'mean',
    'batch_norm': True,
    'batch_norm_alpha': 0.1,
    'drop_prob': DROPOUT_RATE
}

#######################
# Performance Optimization Configuration
#######################

# Data caching options
USE_DATA_CACHE = True
CACHE_DIR = './cache'

# Parallel processing options  
ENABLE_PARALLEL_PROCESSING = True
N_WORKERS = None  # None = use all available CPU cores
USE_THREADS = False

# Memory optimization options
ENABLE_MEMORY_OPTIMIZATION = True
MAX_MEMORY_MB = 4000  # Increased for advanced models
CHUNK_SIZE = 50
OPTIMIZE_DTYPES = True

# Verbose output
VERBOSE_PROCESSING = True
VERBOSE_TRAINING = True

#######################
# Experimental Configuration
#######################

# Quick testing configuration (use limited subjects)
QUICK_TEST_MODE = True
QUICK_TEST_SUBJECTS = 20  # Number of subjects for quick testing (5=最快, 20=平衡, 40=完整)

# Model comparison configuration
COMPARE_MODELS = True
MODELS_TO_COMPARE = [
    'ShallowFBCSPNet',      # Baseline
    'attention_shallow',     # Attention-enhanced baseline
    'transformer',          # Pure transformer
    'hybrid'                # CNN-Transformer hybrid
]

# Cross-validation configuration
USE_CROSS_VALIDATION = False  # For thorough evaluation
CV_FOLDS = 5

#######################
# Results Configuration
#######################

# Metrics to track
TRACK_DETAILED_METRICS = True
METRICS_TO_TRACK = [
    'accuracy', 'precision', 'recall', 'f1_score', 
    'auc', 'confusion_matrix', 'per_class_accuracy'
]

# Visualization options
GENERATE_PLOTS = True
SAVE_MODEL_CHECKPOINTS = True
SAVE_PREDICTIONS = True

#######################
# Backward Compatibility
#######################

# Ensure compatibility with existing code
INPUT_WINDOW_SAMPLES = WINDOW_SIZES[1]  # Default to 1-second windows