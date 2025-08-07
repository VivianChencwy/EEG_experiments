"""
Configuration file for EEG experiments
"""

#######################
# Path Configuration
#######################

# Define paths for the datasets
P3_DATA_DIR = 'd:/Users/vivian/Desktop/UCSD/EEG/P3 Raw Data BIDS-Compatible'
AVO_DATA_DIR = 'd:/Users/vivian/Desktop/UCSD/EEG/ds005863/ds005863'

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
electrode_list = 'common'
# electrode_list = 'all'

# Model Configuration
classifier = 'ShallowFBCSPNet'
# classifier = 'lda'

# Training Configuration
separate_subject_classification = False

# Subject Layer Configuration (only applies to ShallowFBCSPNet + pooled training)
use_subject_layer = True

# Random Seeds for multiple runs
seeds = [1]#, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#######################
# Model Hyperparameters
#######################

# Training parameters
BATCH_SIZE = 32
MAX_EPOCHS = 100
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.001
GAMMA = 1
EARLY_STOPPING_PATIENCE = 10

# Data split parameters
TRAIN_SIZE = 0.6
TEST_SIZE = 0.2
VAL_SIZE = 0.2 

# Preprocessing parameters
TRIAL_START_OFFSET_SAMPLES = 26
TRIAL_STOP_OFFSET_SAMPLES = 154
LOW_FREQ = 0.1
HIGH_FREQ = 30
RESAMPLE_FREQ = 256
INPUT_WINDOW_SAMPLES = 128 