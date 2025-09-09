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
        except Exception:
            # Fallback reference setting
            try:
                if 'cz' in [ch.lower() for ch in raw.ch_names]:
                    raw.set_eeg_reference(['Cz'])
            except Exception:
                pass  # Use original reference

        # Check and convert data units if needed
        raw_data_before = raw.get_data()
        if np.std(raw_data_before) < 1e-6 and np.std(raw_data_before) > 0:
            raw._data *= 1e6  # Convert V to μV
        elif np.std(raw_data_before) == 0:
            raise ValueError("Data is constant or zero")
        
        # Apply filtering and resampling
        raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ)
        raw.resample(RESAMPLE_FREQ)
        
        # Apply filtering and resampling
        raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ)
        raw.resample(RESAMPLE_FREQ)

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
            # Not enough standard events - use all available
            pass  # Use all available standard events
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
        
        # Log balanced dataset info
        print(f"Balanced dataset: {n_selected_oddball} oddball, {n_selected_standard} standard events")

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
        
        # Basic data validation
        if np.any(np.isnan(windows_data)) or np.any(np.isinf(windows_data)):
            raise ValueError("Data contains NaN or infinite values")
        
        print(f"Extracted {len(windows_data)} windows ({np.sum(windows_labels)} oddball, {len(windows_data)-np.sum(windows_labels)} standard)")
        
        # Return custom dataset
        return ManualWindowsDataset(windows_data, windows_labels)
