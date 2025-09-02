"""
Configuration file for EEG experiments
"""

import os

#######################
# Path Configuration
#######################

# Define paths for the datasets
P3_DATA_DIR = '../P3_Raw_Data_BIDS-Compatible'
AVO_DATA_DIR = '../ds005863'

#######################
# Dataset Configuration
#######################

# Option 1: P3 dataset only 
data_dir = P3_DATA_DIR
dataset = 'P3 Raw Data BIDS-Compatible'
use_combined_datasets = False

# Option 2: ds005863 only
# data_dir = AVO_DATA_DIR
# dataset = 'ds005863'
# use_combined_datasets = False

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
classifier = 'ShallowFBCSPNet'
#classifier = 'lda'

# Training Configuration
separate_subject_classification = True

# Subject Layer Configuration (only applies to ShallowFBCSPNet + pooled training)
use_subject_layer = True

# Random Seeds for multiple runs
seeds = [1]#, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#######################
# Model Hyperparameters
#######################

# Training parameters
BATCH_SIZE = 32
MAX_EPOCHS = 1000
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.001
GAMMA = 1
EARLY_STOPPING_PATIENCE = 100

# Model parameters
N_CLASSES = 2

# Data split parameters
TRAIN_SIZE = 0.6
TEST_SIZE = 0.2
VAL_SIZE = 0.2 

# Preprocessing parameters
TRIAL_START_OFFSET_SAMPLES = 0
TRIAL_STOP_OFFSET_SAMPLES = 128
LOW_FREQ = 0.1
HIGH_FREQ = 30
RESAMPLE_FREQ = 256
INPUT_WINDOW_SAMPLES = 128 


#######################
# Optional env overrides
#######################

def _get_env_bool(var_name, default_value):
    value = os.getenv(var_name)
    if value is None:
        return default_value
    value_normalized = value.strip().lower()
    if value_normalized in {"1", "true", "yes", "y", "t"}:
        return True
    if value_normalized in {"0", "false", "no", "n", "f"}:
        return False
    return default_value


def _get_env_str(var_name, default_value):
    value = os.getenv(var_name)
    return default_value if value is None else value


def _get_env_int_list(var_name, default_value):
    value = os.getenv(var_name)
    if value is None:
        return default_value
    try:
        return [int(x.strip()) for x in value.split(',') if x.strip()]
    except Exception:
        return default_value


# Allow overriding dataset roots
P3_DATA_DIR = _get_env_str('P3_DATA_DIR', P3_DATA_DIR)
AVO_DATA_DIR = _get_env_str('AVO_DATA_DIR', AVO_DATA_DIR)

# Dataset selection and mode
dataset = _get_env_str('DATASET', dataset) if 'dataset' in globals() else _get_env_str('DATASET', 'P3 Raw Data BIDS-Compatible')
use_combined_datasets = _get_env_bool('USE_COMBINED_DATASETS', use_combined_datasets) if 'use_combined_datasets' in globals() else _get_env_bool('USE_COMBINED_DATASETS', False)
data_dir = _get_env_str('DATA_DIR', data_dir) if 'data_dir' in globals() else _get_env_str('DATA_DIR', P3_DATA_DIR)

# Experiment configuration
electrode_list = _get_env_str('ELECTRODE_LIST', electrode_list) if 'electrode_list' in globals() else _get_env_str('ELECTRODE_LIST', 'common')
classifier = _get_env_str('CLASSIFIER', classifier) if 'classifier' in globals() else _get_env_str('CLASSIFIER', 'ShallowFBCSPNet')
separate_subject_classification = _get_env_bool('SEPARATE_SUBJECT_CLASSIFICATION', separate_subject_classification) if 'separate_subject_classification' in globals() else _get_env_bool('SEPARATE_SUBJECT_CLASSIFICATION', True)
use_subject_layer = _get_env_bool('USE_SUBJECT_LAYER', use_subject_layer) if 'use_subject_layer' in globals() else _get_env_bool('USE_SUBJECT_LAYER', True)
seeds = _get_env_int_list('SEEDS', seeds) if 'seeds' in globals() else _get_env_int_list('SEEDS', [1])
