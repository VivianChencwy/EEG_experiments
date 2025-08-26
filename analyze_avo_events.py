"""
Analyze AVO dataset events to understand the trial count issue
"""

import os
import numpy as np
import mne
from mne.io import read_raw_brainvision
from eegdash.data_utils import EEGBIDSDataset

def analyze_avo_events():
    """Analyze AVO dataset events to understand the doubling issue."""
    
    print("="*80)
    print("AVO DATASET EVENT ANALYSIS")
    print("="*80)
    
    # Load AVO dataset
    avo_dir = '../ds005863'
    if not os.path.exists(avo_dir):
        print(f"AVO directory not found: {avo_dir}")
        return
    
    try:
        avo_dataset = EEGBIDSDataset(data_dir=avo_dir, dataset='ds005863')
        print(f"AVO dataset loaded successfully")
        
        # Get files for first subject
        all_files = [str(f) for f in avo_dataset.get_files()]
        print(f"Total files in dataset: {len(all_files)}")
        
        # Find files for sub-001
        sub001_files = [f for f in all_files if 'sub-001' in f]
        print(f"Files for sub-001: {len(sub001_files)}")
        
        # Find visual oddball files
        vo_files = [f for f in sub001_files if 'visualoddball' in f]
        print(f"Visual oddball files for sub-001: {len(vo_files)}")
        
        for i, file in enumerate(vo_files):
            print(f"  {i+1}: {os.path.basename(file)}")
        
        # Analyze the first .vhdr file
        vhdr_files = [f for f in vo_files if f.endswith('.vhdr')]
        if not vhdr_files:
            print("No .vhdr files found")
            return
        
        print(f"\nAnalyzing first .vhdr file: {os.path.basename(vhdr_files[0])}")
        
        # Load raw data
        raw = read_raw_brainvision(vhdr_files[0], preload=True)
        print(f"Raw data loaded: {raw.info['nchan']} channels, {raw.info['sfreq']} Hz, {raw.times[-1]:.2f}s")
        
        # Extract events
        events, event_id = mne.events_from_annotations(raw)
        print(f"Total events extracted: {len(events)}")
        print(f"Event ID mapping: {event_id}")
        
        # Show event code distribution
        event_codes = events[:, 2]
        unique_codes, counts = np.unique(event_codes, return_counts=True)
        print(f"\nEvent code distribution:")
        for code, count in zip(unique_codes, counts):
            print(f"  Code {code}: {count} occurrences")
        
        # Show first 20 events
        print(f"\nFirst 20 events (sample, prev_sample, code):")
        for i in range(min(20, len(events))):
            sample, prev_sample, code = events[i]
            time_sec = sample / raw.info['sfreq']
            print(f"  Event {i+1}: sample={sample}, time={time_sec:.3f}s, code={code}")
        
        # Check current constants
        from constants import RESPONSE_EVENTS, ODDBALL_EVENTS
        print(f"\nCurrent constants:")
        print(f"RESPONSE_EVENTS: {RESPONSE_EVENTS}")
        print(f"ODDBALL_EVENTS: {ODDBALL_EVENTS}")
        
        # Apply current filtering
        print(f"\nApplying current filtering:")
        response_mask = np.isin(events[:, 2], RESPONSE_EVENTS)
        events_no_response = events[~response_mask]
        print(f"Events after removing responses: {len(events_no_response)}")
        
        # Remove last event
        events_final = events_no_response[:-1]
        print(f"Events after removing last event: {len(events_final)}")
        
        # Check oddball classification
        oddball_mask = np.isin(events_final[:, 2], ODDBALL_EVENTS)
        oddball_count = np.sum(oddball_mask)
        standard_count = len(events_final) - oddball_count
        
        print(f"Oddball events: {oddball_count}")
        print(f"Standard events: {standard_count}")
        print(f"Total events for windowing: {len(events_final)}")
        
        # Analyze all visual oddball files for this subject
        print(f"\n" + "="*60)
        print("ANALYZING ALL VISUAL ODDBALL FILES FOR SUB-001")
        print("="*60)
        
        total_events_all_files = 0
        for i, vhdr_file in enumerate(vhdr_files):
            print(f"\nFile {i+1}: {os.path.basename(vhdr_file)}")
            raw_file = read_raw_brainvision(vhdr_file, preload=True)
            events_file, _ = mne.events_from_annotations(raw_file)
            
            # Apply filtering
            response_mask = np.isin(events_file[:, 2], RESPONSE_EVENTS)
            events_no_response = events_file[~response_mask]
            events_final_file = events_no_response[:-1] if len(events_no_response) > 0 else events_no_response
            
            print(f"  Total events: {len(events_file)}")
            print(f"  After filtering: {len(events_final_file)}")
            total_events_all_files += len(events_final_file)
        
        print(f"\nTotal events across all files: {total_events_all_files}")
        
        if total_events_all_files > 300:
            print("⚠️  ISSUE CONFIRMED: Total events > 300 (should be ~200)")
            print("This explains why we get ~400 trials instead of ~200")
        else:
            print("✅ Event count looks reasonable")
            
        return {
            'total_files': len(vhdr_files),
            'total_events': total_events_all_files,
            'event_codes': unique_codes,
            'event_counts': counts
        }
        
    except Exception as e:
        print(f"Error analyzing AVO dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = analyze_avo_events()
    
    if results:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Visual oddball files for sub-001: {results['total_files']}")
        print(f"Total events after filtering: {results['total_events']}")
        print(f"Event codes found: {results['event_codes']}")
        
        if results['total_events'] > 300:
            print("\n🔍 PROBLEM IDENTIFIED!")
            print("The AVO dataset has too many events after filtering.")
            print("Need to investigate the correct response event codes for AVO dataset.")
