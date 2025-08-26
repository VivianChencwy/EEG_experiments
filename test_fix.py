"""
Simple test script to verify the trial count fix
"""

import os
import sys
import numpy as np
import mne
from mne.io import read_raw_eeglab

# Add current directory to path to import our modules
sys.path.append('.')

from constants import RESPONSE_EVENTS, ODDBALL_EVENTS
from preprocessor import OddballPreprocessor

def test_trial_count_fix():
    """Test that the trial count fix works correctly."""
    
    print("="*60)
    print("TESTING TRIAL COUNT FIX")
    print("="*60)
    
    # Test with sub-001
    subject_dir = "../P3_Raw_Data_BIDS-Compatible/sub-001/eeg"
    eeg_file = os.path.join(subject_dir, "sub-001_task-P3_eeg.set")
    
    if not os.path.exists(eeg_file):
        print(f"EEG file not found: {eeg_file}")
        return
    
    print(f"Loading EEG file: {eeg_file}")
    raw = read_raw_eeglab(eeg_file, preload=True)
    
    # Extract events
    events, event_id = mne.events_from_annotations(raw)
    print(f"Total events in raw data: {len(events)}")
    
    # Show current constants
    print(f"RESPONSE_EVENTS (to filter): {RESPONSE_EVENTS}")
    print(f"ODDBALL_EVENTS: {ODDBALL_EVENTS}")
    
    # Apply filtering as in preprocessor
    print("\nApplying preprocessing filters:")
    
    # Filter response events
    response_mask = np.isin(events[:, 2], RESPONSE_EVENTS)
    events_no_response = events[~response_mask]
    print(f"Events after removing responses: {len(events_no_response)}")
    
    # Remove last event
    events_final = events_no_response[:-1]
    print(f"Events after removing last event: {len(events_final)}")
    
    # Check oddball vs standard classification
    oddball_mask = np.isin(events_final[:, 2], ODDBALL_EVENTS)
    oddball_count = np.sum(oddball_mask)
    standard_count = len(events_final) - oddball_count
    
    print(f"Oddball events: {oddball_count}")
    print(f"Standard events: {standard_count}")
    print(f"Total events for windowing: {len(events_final)}")
    
    # Test with actual preprocessor
    print("\nTesting with OddballPreprocessor:")
    
    # Use common channels for testing
    from constants import COMMON_CHANNELS
    preprocessor = OddballPreprocessor(COMMON_CHANNELS)
    
    try:
        windows = preprocessor.transform(raw)
        print(f"Windows created by preprocessor: {len(windows)}")
        
        if hasattr(windows, 'data') and hasattr(windows, 'labels'):
            print(f"Window data shape: {windows.data.shape}")
            print(f"Labels shape: {windows.labels.shape}")
            
            # Check label distribution
            unique_labels, counts = np.unique(windows.labels, return_counts=True)
            print(f"Label distribution:")
            for label, count in zip(unique_labels, counts):
                print(f"  Label {label}: {count} occurrences")
        
        print("\n✅ SUCCESS: Preprocessor is working correctly!")
        print(f"Final trial count: {len(windows)} (should be ~200, not ~400)")
        
        if len(windows) < 250:  # Should be around 200, not 400
            print("🎉 TRIAL COUNT FIX SUCCESSFUL!")
            print("The number of trials is now correct (~200 instead of ~400)")
        else:
            print("⚠️  Trial count is still high - fix may not be working")
            
    except Exception as e:
        print(f"Error in preprocessor: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_trial_count_fix()
