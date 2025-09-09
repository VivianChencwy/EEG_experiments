"""
Preprocessor classes for EEG experiments
"""

import numpy as np
import mne
from braindecode.preprocessing import Preprocessor
from braindecode.datasets import BaseConcatDataset, BaseDataset

from constants import RESPONSE_EVENTS, ODDBALL_EVENTS, EVENT_MAPPING
from config import (
    TRIAL_START_OFFSET_SAMPLES, TRIAL_STOP_OFFSET_SAMPLES,
    LOW_FREQ, HIGH_FREQ, RESAMPLE_FREQ
)


class ManualWindowsDataset:
    """Custom dataset that ensures one window per event."""
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class OddballPreprocessor(Preprocessor):
    """Generic preprocessor for oddball-paradigm EEG data."""

    def __init__(self, eeg_channels, 
                 trial_start_offset_samples=TRIAL_START_OFFSET_SAMPLES,
                 trial_stop_offset_samples=TRIAL_STOP_OFFSET_SAMPLES,
                 random_seed=42):
        super().__init__(fn=self.transform, apply_on_array=False)
        self.eeg_channels = [ch.lower() for ch in eeg_channels]
        self.trial_start_offset_samples = trial_start_offset_samples
        self.trial_stop_offset_samples = trial_stop_offset_samples
        self.random_seed = random_seed

    def transform(self, raw):
        """Transform raw EEG data into windowed dataset."""
        # Standardise channel names to lower-case
        raw.rename_channels({ch: ch.lower() for ch in raw.ch_names})

        # Select available channels
        available_channels = [ch for ch in self.eeg_channels if ch in raw.ch_names]
        if not available_channels:
            raise ValueError(
                f"None of the requested channels found. Available: {raw.ch_names}"
            )

        raw.pick_channels(available_channels)
        
        # Set reference to average (common for EEG analysis)
        # This can help if data appears to have zero values due to referencing issues
        try:
            raw.set_eeg_reference('average', projection=True)
            print("Set average reference")
        except Exception as e:
            print(f"Warning: Could not set average reference: {e}")
            # Try setting to specific channels if available
            try:
                if 'cz' in [ch.lower() for ch in raw.ch_names]:
                    raw.set_eeg_reference(['Cz'])
                    print("Set reference to Cz")
                else:
                    print("No reference set - using original reference")
            except Exception as e2:
                print(f"Warning: Could not set any reference: {e2}")

        # Debug: Check raw data before filtering
        raw_data_before = raw.get_data()
        print(f"Raw data shape before filtering: {raw_data_before.shape}")
        print(f"Raw data range before filtering: [{np.min(raw_data_before):.6f}, {np.max(raw_data_before):.6f}]")
        print(f"Raw data std before filtering: {np.std(raw_data_before):.6f}")
        
        # Check if data might be in the wrong units (too small values suggest V instead of μV)
        if np.std(raw_data_before) < 1e-6 and np.std(raw_data_before) > 0:
            print("Data appears to be in Volts, converting to microvolts...")
            raw._data *= 1e6  # Convert V to μV
            raw_data_before = raw.get_data()
            print(f"After unit conversion - range: [{np.min(raw_data_before):.6f}, {np.max(raw_data_before):.6f}]")
            print(f"After unit conversion - std: {np.std(raw_data_before):.6f}")
        elif np.std(raw_data_before) == 0:
            print("ERROR: All data values are zero or constant!")
        
        # Apply filtering and resampling
        raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ)
        raw.resample(RESAMPLE_FREQ)
        
        # Debug: Check raw data after filtering
        raw_data_after = raw.get_data()
        print(f"Raw data shape after filtering: {raw_data_after.shape}")
        print(f"Raw data range after filtering: [{np.min(raw_data_after):.6f}, {np.max(raw_data_after):.6f}]")
        print(f"Raw data std after filtering: {np.std(raw_data_after):.6f}")

        # Extract events
        events, _ = mne.events_from_annotations(raw)
        if len(events) == 0:
            raise ValueError("No events found after reading annotations.")

        # Drop response events first
        response_mask = np.isin(events[:, 2], RESPONSE_EVENTS)
        events = events[~response_mask]
        if len(events) == 0:
            raise ValueError("No non-response events found after filtering.")

        # Remove last remaining (non-response) event to avoid trailing window overflow
        events = events[:-1]
        

        # Separate oddball and standard events for balanced sampling
        oddball_mask = np.isin(events[:, 2], ODDBALL_EVENTS)
        oddball_events = events[oddball_mask]
        standard_events = events[~oddball_mask]
        
        # Balance the dataset by using all oddball events and randomly sampling standard events
        n_oddball = len(oddball_events)
        n_standard = len(standard_events)
        
        if n_oddball == 0:
            raise ValueError("No oddball events found in the data.")
        if n_standard == 0:
            raise ValueError("No standard events found in the data.")
        
        # Use all oddball events
        selected_oddball_events = oddball_events.copy()
        
        # Set random seed for reproducible sampling
        np.random.seed(self.random_seed)
        
        # Randomly sample standard events to match oddball count
        if n_standard >= n_oddball:
            # Enough standard events - randomly sample without replacement
            standard_indices = np.random.choice(n_standard, size=n_oddball, replace=False)
            selected_standard_events = standard_events[standard_indices]
        else:
            # Not enough standard events - use all and warn
            print(f"Warning: Only {n_standard} standard events available, but {n_oddball} oddball events. Using all standard events.")
            selected_standard_events = standard_events.copy()
        
        # Combine selected events and create labels
        selected_events = np.vstack([selected_oddball_events, selected_standard_events])
        
        # Create balanced labels (1 for oddball, 0 for standard)
        n_selected_oddball = len(selected_oddball_events)
        n_selected_standard = len(selected_standard_events)
        labels = np.concatenate([
            np.ones(n_selected_oddball, dtype=int),  # oddball = 1
            np.zeros(n_selected_standard, dtype=int)  # standard = 0
        ])
        
        print(f"Balanced dataset: {n_selected_oddball} oddball events, {n_selected_standard} standard events")
        
        # Debug: Check that events are properly distributed
        print(f"Original distribution - Oddball: {n_oddball}, Standard: {n_standard}")
        print(f"Selected oddball indices: {np.sort(selected_oddball_events[:, 0])[:5]} ... (showing first 5)")
        if n_standard >= n_oddball:
            print(f"Selected standard indices: {np.sort(selected_standard_events[:, 0])[:5]} ... (showing first 5)")
        print(f"Event codes in selected oddball: {np.unique(selected_oddball_events[:, 2])}")
        print(f"Event codes in selected standard: {np.unique(selected_standard_events[:, 2])}")

        # Manual window extraction to ensure one window per event
        raw_data = raw.get_data()  # Shape: (n_channels, n_timepoints)
        sfreq = raw.info['sfreq']
        
        # Extract windows manually
        windows_data = []
        windows_labels = []
        
        window_size = self.trial_stop_offset_samples - self.trial_start_offset_samples
        
        for i, (event_sample, _, _) in enumerate(selected_events):
            # Calculate window boundaries
            start_sample = event_sample + self.trial_start_offset_samples
            end_sample = event_sample + self.trial_stop_offset_samples
            
            # Check if window is within data bounds
            if start_sample >= 0 and end_sample <= raw_data.shape[1]:
                # Extract window data
                window_data = raw_data[:, start_sample:end_sample]  # Shape: (n_channels, window_size)
                
                # Store window and label
                windows_data.append(window_data)
                windows_labels.append(labels[i])
        
        # Convert to numpy arrays
        windows_data = np.array(windows_data)  # Shape: (n_windows, n_channels, window_size)
        windows_labels = np.array(windows_labels)  # Shape: (n_windows,)
        
        # Data quality checks
        print(f"Final dataset shape: {windows_data.shape}")
        print(f"Final labels shape: {windows_labels.shape}")
        print(f"Final label distribution: {np.bincount(windows_labels)}")
        print(f"Data range: [{np.min(windows_data):.6f}, {np.max(windows_data):.6f}]")
        print(f"Data std: {np.std(windows_data):.6f}")
        
        # Check for any NaN or infinite values
        if np.any(np.isnan(windows_data)) or np.any(np.isinf(windows_data)):
            print("WARNING: NaN or infinite values found in data!")
        
        # Return custom dataset
        return ManualWindowsDataset(windows_data, windows_labels)
