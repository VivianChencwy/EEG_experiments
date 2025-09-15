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

#classifier = 'EEGNet'
#classifier = 'EEGConformer'
classifier = 'ShallowFBCSPNet'
#classifier = 'DeepConvNet' #problem
#classifier = 'EEGChannelNet'
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
TRIAL_START_OFFSET_SAMPLES = 0
TRIAL_STOP_OFFSET_SAMPLES = int(1.0 * 128)  # 1 second at 128 Hz

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

# Alternative: Fixed trial counts (if you want exact numbers instead of ratios)
# Set these to specific numbers if you want exact trial counts
FIXED_TRIALS_PER_SUBJECT_TRAIN = 10  # e.g., 100 for exactly 100 train trials per subject
FIXED_TRIALS_PER_SUBJECT_VAL = 10    # e.g., 20 for exactly 20 val trials per subject
FIXED_TRIALS_PER_SUBJECT_TEST = 10   # e.g., 30 for exactly 30 test trials per subject

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
# Model Configuration Details
#######################

# Input/Output dimensions
INPUT_WINDOW_SAMPLES = int(1.0 * 128)  # 1 second at 128 Hz
N_CLASSES = 2

# Training hyperparameters
LEARNING_RATE = 0.0005  # Reduced for better convergence
WEIGHT_DECAY = 1e-5     # Reduced weight decay
GAMMA = 0.5  # Learning rate decay factor
EARLY_STOPPING_PATIENCE = 30 # Increased patience for complex models
DROPOUT_RATE = 0.3      # Reduced dropout for transformer models

# Data augmentation (enabled for better generalization)
USE_DATA_AUGMENTATION = True
NOISE_STD = 0.005       # Reduced noise
TIME_SHIFT_RANGE = 5    # Small time shifts (in samples)
LABEL_SMOOTHING = 0.05  # Reduced label smoothing

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