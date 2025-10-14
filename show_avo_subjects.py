#!/usr/bin/env python3
"""
Simple script to show the 40 AVO subjects selected based on oddball trial counts.
This script extracts the subject selection logic from the main experiment code.
"""

import os
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import AVO_DATA_DIR
from experiment import get_dataset_subjects
from data_utils import EEGBIDSDataset

def main():
    print("=" * 60)
    print("AVO Dataset Subject Selection")
    print("=" * 60)
    print(f"AVO Data Directory: {AVO_DATA_DIR}")
    print()
    
    try:
        # Create AVO dataset object
        print("Loading AVO dataset...")
        avo_dataset = EEGBIDSDataset(data_dir=AVO_DATA_DIR, dataset='ds005863')
        
        # Get the selected subjects (this will automatically select top 40 based on oddball counts)
        print("Selecting subjects based on oddball trial counts...")
        print()
        
        subjects = get_dataset_subjects('AVO', avo_dataset)
        
        print("=" * 60)
        print(f"SELECTED {len(subjects)} AVO SUBJECTS")
        print("=" * 60)
        print("The following subjects were selected based on having the most oddball trials:")
        print()
        
        for i, subject_id in enumerate(subjects, 1):
            print(f"{i:2d}. sub-{subject_id}")
        
        print()
        print("=" * 60)
        print("Subject selection complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the AVO dataset is available and the conda environment is activated.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
