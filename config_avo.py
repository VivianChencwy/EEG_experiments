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
#electrode_list = 'common'
electrode_list = 'all'

# Model Configuration
#classifier = 'ShallowFBCSPNet'
classifier = 'lda'

# Training Configuration
separate_subject_classification = True

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
MAX_EPOCHS = 200

# Dataset split ratios (must sum to ≤ 1.0)
TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
TEST_SIZE = 0.2

# Random seeds for reproducibility
seeds = [42, 123, 456, 789, 321]