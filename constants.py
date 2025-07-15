"""
Constants for EEG experiments
"""

#######################
# Channel Constants
#######################

COMMON_CHANNELS = [
    'FP1', 'Fz', 'F3', 'F7', 'C3', 'P3', 'P7', 'O1', 
    'Oz', 'Pz', 'O2', 'P4', 'P8', 'C4', 'F4', 'F8'
]

P3_CHANNELS = [
    'FP1', 'F3', 'F7', 'FC3', 'C3', 'C5', 'P3', 'P7', 'P9', 'PO7', 
    'PO3', 'O1', 'Oz', 'Pz', 'CPz', 'FP2', 'Fz', 'F4', 'F8', 'FC4', 
    'FCz', 'Cz', 'C4', 'C6', 'P4', 'P8', 'P10', 'PO8', 'PO4', 'O2'
]

AVO_CHANNELS = [
    'Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 
    'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 
    'P4', 'P8', 'CP6', 'CP2', 'C4', 'FC6', 'FC2', 'F4', 
    'F8', 'Fp2'
]

#######################
# Event Constants
#######################

# Response event codes to remove
RESPONSE_EVENTS = [201, 202]

# Oddball event codes
ODDBALL_EVENTS = [11, 22, 33, 44, 55]

# Event mapping
EVENT_MAPPING = {0: "standard", 1: "oddball"}

#######################
# Data Processing Constants
#######################

# Normalization epsilon for numerical stability
NORMALIZATION_EPSILON = 1e-7 