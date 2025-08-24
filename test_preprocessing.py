#!/usr/bin/env python3
"""
Quick test script to verify the preprocessing fix
"""

import os
import pandas as pd
from preprocessor import OddballPreprocessor
from constants import COMMON_CHANNELS
from utils import load_raw

def test_single_subject():
    """Test preprocessing on a single subject"""
    print("Testing preprocessing fix...")
    
    # Test with one subject
    subject_id = 'sub-001'
    dataset_dir = 'P3_Raw_Data_BIDS-Compatible'
    events_file = os.path.join(dataset_dir, subject_id, 'eeg', f'{subject_id}_task-P3_events.tsv')
    eeg_file = os.path.join(dataset_dir, subject_id, 'eeg', f'{subject_id}_task-P3_eeg.set')
    
    if not os.path.exists(events_file) or not os.path.exists(eeg_file):
        print(f"Files not found for {subject_id}")
        return False
    
    try:
        # Load data
        print(f"Loading events from {events_file}")
        events_df = pd.read_csv(events_file, sep='\t')
        print(f"Found {len(events_df)} events")
        
        print(f"Loading EEG data from {eeg_file}")
        raw = load_raw(eeg_file, 'P3')
        raw.load_data()
        print(f"EEG data loaded: {raw.info['nchan']} channels, {raw.info['sfreq']} Hz")
        
        # Test preprocessor
        print("Creating preprocessor...")
        preprocessor = OddballPreprocessor(COMMON_CHANNELS)
        
        print("Running preprocessing...")
        windows, event_counts = preprocessor.transform(raw, events_df)
        
        print(f"SUCCESS! Created {len(windows)} windows")
        print(f"Event counts: {event_counts}")
        
        # Test accessing the data
        print("Testing data access...")
        sample_data, sample_label = windows[0]
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Sample label: {sample_label}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_subject()
    if success:
        print("\n✓ Preprocessing fix successful!")
    else:
        print("\n✗ Preprocessing fix failed!")
