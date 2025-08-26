"""
Data consistency analysis script for sub-001
This script analyzes the actual EEG data and compares it with TSV events
"""

import os
import numpy as np
import pandas as pd
import mne
from mne.io import read_raw_eeglab

def analyze_sub001_data():
    """Analyze sub-001 data consistency between EEG file and TSV file."""
    
    # Paths
    subject_dir = "../P3_Raw_Data_BIDS-Compatible/sub-001/eeg"
    eeg_file = os.path.join(subject_dir, "sub-001_task-P3_eeg.set")
    tsv_file = os.path.join(subject_dir, "sub-001_task-P3_events.tsv")
    
    print("="*80)
    print("DATA CONSISTENCY ANALYSIS FOR SUB-001")
    print("="*80)
    
    # 1. Load and analyze TSV events file
    print("\n1. ANALYZING TSV EVENTS FILE:")
    print("-" * 40)
    
    if os.path.exists(tsv_file):
        events_df = pd.read_csv(tsv_file, sep='\t')
        print(f"TSV file loaded successfully: {tsv_file}")
        print(f"Total rows in TSV: {len(events_df)}")
        print(f"Columns: {list(events_df.columns)}")
        
        # Filter stimulus events
        stimulus_events = events_df[events_df['trial_type'] == 'stimulus']
        print(f"Stimulus events in TSV: {len(stimulus_events)}")
        
        # Show event value distribution
        if 'value' in stimulus_events.columns:
            value_counts = stimulus_events['value'].value_counts().sort_index()
            print(f"Event value distribution in TSV:")
            for value, count in value_counts.items():
                print(f"  Value {value}: {count} occurrences")
            
            # Show first 20 event values
            event_values = stimulus_events['value'].tolist()
            print(f"First 20 event values: {event_values[:20]}")
            print(f"Last 20 event values: {event_values[-20:]}")
        
        # Check for response events
        if 'trial_type' in events_df.columns:
            trial_type_counts = events_df['trial_type'].value_counts()
            print(f"Trial type distribution:")
            for trial_type, count in trial_type_counts.items():
                print(f"  {trial_type}: {count} occurrences")
    else:
        print(f"TSV file not found: {tsv_file}")
        return
    
    # 2. Load and analyze raw EEG data
    print("\n2. ANALYZING RAW EEG DATA:")
    print("-" * 40)
    
    if os.path.exists(eeg_file):
        print(f"Loading EEG file: {eeg_file}")
        raw = read_raw_eeglab(eeg_file, preload=True)
        print(f"EEG data loaded successfully")
        print(f"Sampling frequency: {raw.info['sfreq']} Hz")
        print(f"Duration: {raw.times[-1]:.2f} seconds")
        print(f"Number of channels: {len(raw.ch_names)}")
        
        # Extract events from annotations
        events, event_id = mne.events_from_annotations(raw)
        print(f"Total events extracted from EEG: {len(events)}")
        print(f"Event ID mapping: {event_id}")
        
        # Show event code distribution
        event_codes = events[:, 2]
        unique_codes, counts = np.unique(event_codes, return_counts=True)
        print(f"Event code distribution in EEG:")
        for code, count in zip(unique_codes, counts):
            print(f"  Code {code}: {count} occurrences")
        
        # Show first and last 20 events with timestamps
        print(f"First 20 events (sample, prev_sample, code):")
        for i in range(min(20, len(events))):
            sample, prev_sample, code = events[i]
            time_sec = sample / raw.info['sfreq']
            print(f"  Event {i+1}: sample={sample}, time={time_sec:.3f}s, code={code}")
        
        print(f"Last 20 events:")
        for i in range(max(0, len(events)-20), len(events)):
            sample, prev_sample, code = events[i]
            time_sec = sample / raw.info['sfreq']
            print(f"  Event {i+1}: sample={sample}, time={time_sec:.3f}s, code={code}")
        
    else:
        print(f"EEG file not found: {eeg_file}")
        return
    
    # 3. Compare TSV and EEG events
    print("\n3. COMPARING TSV AND EEG EVENTS:")
    print("-" * 40)
    
    # Map EEG event codes to stimulus values if possible
    # This depends on the specific mapping used in the dataset
    print(f"TSV stimulus events: {len(stimulus_events)}")
    print(f"EEG total events: {len(events)}")
    
    # Try to identify response events in EEG data
    # Common response event codes might be higher numbers
    potential_response_codes = []
    potential_stimulus_codes = []
    
    for code, count in zip(unique_codes, counts):
        # Heuristic: response events might be less frequent or have specific codes
        if count < len(stimulus_events) * 0.8:  # Less than 80% of stimulus events
            potential_response_codes.append(code)
        else:
            potential_stimulus_codes.append(code)
    
    print(f"Potential stimulus codes: {potential_stimulus_codes}")
    print(f"Potential response codes: {potential_response_codes}")
    
    # Filter out potential response events
    stimulus_mask = np.isin(events[:, 2], potential_stimulus_codes)
    stimulus_events_eeg = events[stimulus_mask]
    print(f"EEG stimulus events (after filtering): {len(stimulus_events_eeg)}")
    
    # 4. Detailed analysis of discrepancies
    print("\n4. DISCREPANCY ANALYSIS:")
    print("-" * 40)
    
    tsv_count = len(stimulus_events)
    eeg_count = len(stimulus_events_eeg)
    
    print(f"TSV stimulus events: {tsv_count}")
    print(f"EEG stimulus events: {eeg_count}")
    print(f"Difference: {eeg_count - tsv_count}")
    
    if tsv_count != eeg_count:
        print("⚠️  INCONSISTENCY DETECTED!")
        print("The number of stimulus events in TSV and EEG files do not match.")
        
        if eeg_count > tsv_count:
            print(f"EEG has {eeg_count - tsv_count} more events than TSV")
            print("This could explain why we get ~400 trials instead of ~200")
        else:
            print(f"TSV has {tsv_count - eeg_count} more events than EEG")
    else:
        print("✅ Event counts match between TSV and EEG")
    
    # 5. Simulate preprocessing to see actual window count
    print("\n5. SIMULATING PREPROCESSING:")
    print("-" * 40)
    
    # Apply the same filtering as in the preprocessor
    from constants import RESPONSE_EVENTS, ODDBALL_EVENTS
    
    print(f"Response event codes to filter: {RESPONSE_EVENTS}")
    print(f"Oddball event codes: {ODDBALL_EVENTS}")
    
    # Filter response events
    response_mask = np.isin(events[:, 2], RESPONSE_EVENTS)
    events_no_response = events[~response_mask]
    print(f"Events after removing responses: {len(events_no_response)}")
    
    # Remove last event (as done in preprocessor)
    events_final = events_no_response[:-1]
    print(f"Events after removing last event: {len(events_final)}")
    
    print(f"Final event codes distribution:")
    final_codes = events_final[:, 2]
    unique_final, counts_final = np.unique(final_codes, return_counts=True)
    for code, count in zip(unique_final, counts_final):
        print(f"  Code {code}: {count} occurrences")
    
    # Check which codes are mapped to oddball vs standard
    oddball_mask = np.isin(events_final[:, 2], ODDBALL_EVENTS)
    oddball_count = np.sum(oddball_mask)
    standard_count = len(events_final) - oddball_count
    
    print(f"Oddball events: {oddball_count}")
    print(f"Standard events: {standard_count}")
    print(f"Total events for windowing: {len(events_final)}")
    
    return {
        'tsv_stimulus_count': tsv_count,
        'eeg_total_count': len(events),
        'eeg_stimulus_count': eeg_count,
        'final_window_count': len(events_final),
        'oddball_count': oddball_count,
        'standard_count': standard_count
    }

if __name__ == "__main__":
    results = analyze_sub001_data()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if results:
        print(f"TSV stimulus events: {results['tsv_stimulus_count']}")
        print(f"EEG total events: {results['eeg_total_count']}")
        print(f"EEG stimulus events: {results['eeg_stimulus_count']}")
        print(f"Final window count: {results['final_window_count']}")
        print(f"Oddball events: {results['oddball_count']}")
        print(f"Standard events: {results['standard_count']}")
        
        if results['final_window_count'] == 400:
            print("\n🔍 FOUND THE ISSUE!")
            print("The final window count is indeed 400, which explains the doubled trials.")
            print("This suggests the original data actually contains ~400 stimulus events,")
            print("not ~200 as might be expected from the TSV file.")
