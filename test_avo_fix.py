#!/usr/bin/env python3
"""
Test script to verify AVO dataset processing fix
"""

import os
import sys
import shutil
sys.path.append('.')

from config import AVO_DATA_DIR
from data_utils import EEGBIDSDataset
from constants_avo import AVO_CHANNELS
from preprocessor import OddballPreprocessor
from utils import process_subject_data
from experiment_logger import setup_logger

def test_avo_processing():
    print("=== Testing AVO Dataset Processing ===")
    
    # Clean cache first
    cache_dir = './cache'
    if os.path.exists(cache_dir):
        print("Cleaning cache directory...")
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)
    
    # Setup logger
    logger = setup_logger("test_avo", "test_avo.log")
    
    # Initialize dataset
    dataset = EEGBIDSDataset(AVO_DATA_DIR)
    print(f"Dataset: {dataset}")
    
    # Test with a few subjects
    test_subjects = ['001', '002', '005', '008']
    
    for subject_id in test_subjects:
        print(f"\n--- Testing Subject sub-{subject_id} ---")
        
        try:
            # Create preprocessor with AVO dataset type
            preprocessor = OddballPreprocessor(AVO_CHANNELS, dataset_type='AVO')
            
            # Process subject data
            data, labels = process_subject_data(subject_id, dataset, preprocessor, logger, dataset_type='AVO')
            
            if data is not None and labels is not None:
                print(f"  Success: {len(data)} windows, {len(labels)} labels")
                print(f"  Data shape: {data.shape}")
                print(f"  Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
            else:
                print(f"  Failed: No data returned")
                
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    import numpy as np
    test_avo_processing()
