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
                 trial_stop_offset_samples=TRIAL_STOP_OFFSET_SAMPLES):
        super().__init__(fn=self.transform, apply_on_array=False)
        self.eeg_channels = [ch.lower() for ch in eeg_channels]
        self.trial_start_offset_samples = trial_start_offset_samples
        self.trial_stop_offset_samples = trial_stop_offset_samples

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
        

        # Map oddball codes → 1, standard → 0
        oddball_mask = np.isin(events[:, 2], ODDBALL_EVENTS)
        new_events = np.zeros_like(events)
        new_events[:, 0] = events[:, 0]
        new_events[oddball_mask, 2] = 1

        # Manual window extraction to ensure one window per event
        raw_data = raw.get_data()  # Shape: (n_channels, n_timepoints)
        sfreq = raw.info['sfreq']
        
        # Extract windows manually
        windows_data = []
        windows_labels = []
        
        window_size = self.trial_stop_offset_samples - self.trial_start_offset_samples
        
        for event_sample, _, event_code in new_events:
            # Calculate window boundaries
            start_sample = event_sample + self.trial_start_offset_samples
            end_sample = event_sample + self.trial_stop_offset_samples
            
            # Check if window is within data bounds
            if start_sample >= 0 and end_sample <= raw_data.shape[1]:
                # Extract window data
                window_data = raw_data[:, start_sample:end_sample]  # Shape: (n_channels, window_size)
                
                # Store window and label
                windows_data.append(window_data)
                windows_labels.append(event_code)
        
        # Convert to numpy arrays
        windows_data = np.array(windows_data)  # Shape: (n_windows, n_channels, window_size)
        windows_labels = np.array(windows_labels)  # Shape: (n_windows,)
        
        
        # Return custom dataset
        return ManualWindowsDataset(windows_data, windows_labels)
