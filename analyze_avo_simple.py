"""
Simple AVO dataset event analysis without eegdash dependency
"""

import os
import numpy as np
import mne
from mne.io import read_raw_brainvision

def analyze_avo_simple():
    """Analyze AVO dataset events directly from files."""
    
    print("="*80)
    print("AVO DATASET SIMPLE EVENT ANALYSIS")
    print("="*80)
    
    # Direct path to AVO dataset
    avo_dir = '../ds005863'
    if not os.path.exists(avo_dir):
        print(f"AVO directory not found: {avo_dir}")
        return
    
    # Look for sub-001 visual oddball files
    sub001_dir = os.path.join(avo_dir, 'sub-001', 'eeg')
    if not os.path.exists(sub001_dir):
        print(f"Sub-001 directory not found: {sub001_dir}")
        return
    
    print(f"Looking in: {sub001_dir}")
    
    # Find all files for sub-001
    all_files = os.listdir(sub001_dir)
    print(f"All files in sub-001/eeg: {len(all_files)}")
    
    # Find visual oddball .vhdr files
    vhdr_files = [f for f in all_files if 'visualoddball' in f and f.endswith('.vhdr')]
    print(f"Visual oddball .vhdr files: {len(vhdr_files)}")
    
    for i, file in enumerate(vhdr_files):
        print(f"  {i+1}: {file}")
    
    if not vhdr_files:
        print("No visual oddball .vhdr files found")
        return
    
    # Analyze each file
    total_events_all_files = 0
    all_event_codes = []
    
    for i, vhdr_file in enumerate(vhdr_files):
        print(f"\n" + "="*60)
        print(f"ANALYZING FILE {i+1}: {vhdr_file}")
        print("="*60)
        
        file_path = os.path.join(sub001_dir, vhdr_file)
        
        try:
            # Load raw data
            raw = read_raw_brainvision(file_path, preload=True)
            print(f"Raw data loaded: {raw.info['nchan']} channels, {raw.info['sfreq']} Hz, {raw.times[-1]:.2f}s")
            
            # Extract events
            events, event_id = mne.events_from_annotations(raw)
            print(f"Total events extracted: {len(events)}")
            print(f"Event ID mapping: {event_id}")
            
            # Show event code distribution
            event_codes = events[:, 2]
            unique_codes, counts = np.unique(event_codes, return_counts=True)
            print(f"Event code distribution:")
            for code, count in zip(unique_codes, counts):
                print(f"  Code {code}: {count} occurrences")
            
            all_event_codes.extend(event_codes)
            
            # Show first 10 events
            print(f"First 10 events (sample, prev_sample, code):")
            for j in range(min(10, len(events))):
                sample, prev_sample, code = events[j]
                time_sec = sample / raw.info['sfreq']
                print(f"  Event {j+1}: sample={sample}, time={time_sec:.3f}s, code={code}")
            
            # Apply current filtering
            from constants import RESPONSE_EVENTS, ODDBALL_EVENTS
            print(f"\nApplying current filtering:")
            print(f"RESPONSE_EVENTS to filter: {RESPONSE_EVENTS}")
            print(f"ODDBALL_EVENTS: {ODDBALL_EVENTS}")
            
            response_mask = np.isin(events[:, 2], RESPONSE_EVENTS)
            events_no_response = events[~response_mask]
            print(f"Events after removing responses: {len(events_no_response)}")
            
            # Remove last event
            events_final = events_no_response[:-1] if len(events_no_response) > 0 else events_no_response
            print(f"Events after removing last event: {len(events_final)}")
            
            total_events_all_files += len(events_final)
            
            # Check oddball classification
            oddball_mask = np.isin(events_final[:, 2], ODDBALL_EVENTS)
            oddball_count = np.sum(oddball_mask)
            standard_count = len(events_final) - oddball_count
            
            print(f"Oddball events: {oddball_count}")
            print(f"Standard events: {standard_count}")
            
        except Exception as e:
            print(f"Error analyzing file {vhdr_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Overall summary
    print(f"\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(f"Total visual oddball files: {len(vhdr_files)}")
    print(f"Total events after filtering: {total_events_all_files}")
    
    # Show overall event code distribution
    all_event_codes = np.array(all_event_codes)
    unique_all_codes, counts_all = np.unique(all_event_codes, return_counts=True)
    print(f"\nOverall event code distribution:")
    for code, count in zip(unique_all_codes, counts_all):
        print(f"  Code {code}: {count} occurrences")
    
    if total_events_all_files > 300:
        print(f"\n⚠️  ISSUE CONFIRMED: Total events = {total_events_all_files} (should be ~200)")
        print("This explains why we get ~400 trials instead of ~200")
        
        # Suggest potential response event codes
        print(f"\nPOTENTIAL RESPONSE EVENT CODES TO INVESTIGATE:")
        # Look for codes that might be response events (typically higher frequency)
        for code, count in zip(unique_all_codes, counts_all):
            if count > 50:  # Codes that appear frequently might include responses
                print(f"  Code {code}: {count} occurrences (potential response events?)")
    else:
        print("✅ Event count looks reasonable")
    
    return {
        'total_files': len(vhdr_files),
        'total_events': total_events_all_files,
        'event_codes': unique_all_codes,
        'event_counts': counts_all
    }

if __name__ == "__main__":
    results = analyze_avo_simple()
