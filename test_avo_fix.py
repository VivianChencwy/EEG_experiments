"""
Test script to verify the AVO dataset trial count fix
"""

import os
import numpy as np
import mne
from mne.io import read_raw_brainvision

def test_avo_fix():
    """Test that the AVO trial count fix works correctly."""
    
    print("="*60)
    print("TESTING AVO TRIAL COUNT FIX")
    print("="*60)
    
    # Test with AVO sub-001
    avo_dir = '../ds005863'
    sub001_dir = os.path.join(avo_dir, 'sub-001', 'eeg')
    
    if not os.path.exists(sub001_dir):
        print(f"AVO sub-001 directory not found: {sub001_dir}")
        return
    
    # Find visual oddball .vhdr files
    all_files = os.listdir(sub001_dir)
    vhdr_files = [f for f in all_files if 'visualoddball' in f and f.endswith('.vhdr')]
    
    if not vhdr_files:
        print("No visual oddball .vhdr files found")
        return
    
    print(f"Found {len(vhdr_files)} visual oddball files")
    
    # Load the updated constants
    from constants import RESPONSE_EVENTS, ODDBALL_EVENTS
    print(f"Updated RESPONSE_EVENTS: {RESPONSE_EVENTS}")
    print(f"ODDBALL_EVENTS: {ODDBALL_EVENTS}")
    
    total_events_before = 0
    total_events_after = 0
    
    for vhdr_file in vhdr_files:
        print(f"\nProcessing: {vhdr_file}")
        file_path = os.path.join(sub001_dir, vhdr_file)
        
        try:
            # Load raw data
            raw = read_raw_brainvision(file_path, preload=True)
            
            # Extract events
            events, _ = mne.events_from_annotations(raw)
            print(f"  Total events: {len(events)}")
            total_events_before += len(events)
            
            # Apply filtering with updated constants
            response_mask = np.isin(events[:, 2], RESPONSE_EVENTS)
            events_no_response = events[~response_mask]
            print(f"  After removing responses: {len(events_no_response)}")
            
            # Remove last event
            events_final = events_no_response[:-1] if len(events_no_response) > 0 else events_no_response
            print(f"  Final events for windowing: {len(events_final)}")
            total_events_after += len(events_final)
            
            # Check oddball vs standard classification
            oddball_mask = np.isin(events_final[:, 2], ODDBALL_EVENTS)
            oddball_count = np.sum(oddball_mask)
            standard_count = len(events_final) - oddball_count
            
            print(f"  Oddball events: {oddball_count}")
            print(f"  Standard events: {standard_count}")
            
        except Exception as e:
            print(f"  Error processing {vhdr_file}: {e}")
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total events before filtering: {total_events_before}")
    print(f"Total events after filtering: {total_events_after}")
    print(f"Events removed: {total_events_before - total_events_after}")
    
    if total_events_after < 250:  # Should be around 200, not 400+
        print("\n🎉 SUCCESS: AVO trial count fix is working!")
        print(f"Final trial count: {total_events_after} (should be ~200)")
        print("The response events (201, 202) are now being properly filtered out.")
    else:
        print(f"\n⚠️  Trial count is still high: {total_events_after}")
        print("The fix may not be working correctly.")
    
    # Test with actual preprocessor
    print(f"\n" + "="*40)
    print("TESTING WITH PREPROCESSOR")
    print("="*40)
    
    try:
        from constants import COMMON_CHANNELS
        from preprocessor import OddballPreprocessor
        
        # Use first file for testing
        file_path = os.path.join(sub001_dir, vhdr_files[0])
        raw = read_raw_brainvision(file_path, preload=True)
        
        preprocessor = OddballPreprocessor(COMMON_CHANNELS)
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
        
        if len(windows) < 250:
            print("\n✅ PREPROCESSOR TEST SUCCESSFUL!")
            print(f"Preprocessor creates {len(windows)} windows (should be ~200)")
        else:
            print(f"\n⚠️  Preprocessor still creates too many windows: {len(windows)}")
            
    except Exception as e:
        print(f"Error testing preprocessor: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_avo_fix()
