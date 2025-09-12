"""
Preprocessor classes for EEG experiments
"""

import numpy as np
import mne
from braindecode.preprocessing import Preprocessor
from braindecode.datasets import BaseConcatDataset, BaseDataset

from constants import RESPONSE_EVENTS, ODDBALL_EVENTS, EVENT_MAPPING
from constants_avo import RESPONSE_EVENTS_AVO, ODDBALL_EVENTS_AVO
from config import (
    TRIAL_START_OFFSET_SAMPLES, TRIAL_STOP_OFFSET_SAMPLES,
    LOW_FREQ, HIGH_FREQ, RESAMPLE_FREQ, FIXED_TRIALS_PER_CLASS,
    TRAIN_TRIALS_PER_CLASS, VAL_TRIALS_PER_CLASS, TEST_TRIALS_PER_CLASS
)
from data_cache import EEGDataCache


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
                 random_seed=42,
                 use_cache=True,
                 dataset_type='P3',
                 fixed_trials_per_class=FIXED_TRIALS_PER_CLASS,
                 use_fixed_split=True):
        super().__init__(fn=self.transform, apply_on_array=False)
        self.eeg_channels = [ch.lower() for ch in eeg_channels]
        self.trial_start_offset_samples = trial_start_offset_samples
        self.trial_stop_offset_samples = trial_stop_offset_samples
        self.random_seed = random_seed
        self.use_cache = use_cache
        self.dataset_type = dataset_type
        self.fixed_trials_per_class = fixed_trials_per_class
        self.use_fixed_split = use_fixed_split
        self.cache = EEGDataCache() if use_cache else None
        
        # Set event codes based on dataset type
        if dataset_type == 'AVO':
            self.response_events = RESPONSE_EVENTS_AVO
            self.oddball_events = ODDBALL_EVENTS_AVO
        else:  # P3 or default
            self.response_events = RESPONSE_EVENTS
            self.oddball_events = ODDBALL_EVENTS

    def transform(self, raw):
        """Transform raw EEG data into windowed dataset."""
        # Check cache first if enabled
        if self.use_cache and self.cache is not None:
            # Try to get raw file path from the raw object
            raw_file = getattr(raw, 'filenames', ['unknown'])[0] if hasattr(raw, 'filenames') else 'unknown'
            
            # Check for cached data
            cached_result = self.cache.get_cached_data(
                raw_file=raw_file,
                channels=self.eeg_channels,
                trial_start_offset=self.trial_start_offset_samples,
                trial_stop_offset=self.trial_stop_offset_samples,
                low_freq=LOW_FREQ,
                high_freq=HIGH_FREQ,
                resample_freq=RESAMPLE_FREQ
            )
            
            if cached_result is not None:
                windows_data, windows_labels = cached_result
                return ManualWindowsDataset(windows_data, windows_labels)
        
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

        # Extract events
        events, _ = mne.events_from_annotations(raw)
        if len(events) == 0:
            raise ValueError("No events found after reading annotations.")

        # Drop response events first
        response_mask = np.isin(events[:, 2], self.response_events)
        events = events[~response_mask]
        if len(events) == 0:
            raise ValueError("No non-response events found after filtering.")

        # Remove last remaining (non-response) event to avoid trailing window overflow
        events = events[:-1]
        

        # Separate oddball and standard events for balanced sampling
        oddball_mask = np.isin(events[:, 2], self.oddball_events)
        oddball_events = events[oddball_mask]
        standard_events = events[~oddball_mask]
        
        # Use fixed number of trials per class
        n_oddball = len(oddball_events)
        n_standard = len(standard_events)
        
        if n_oddball == 0:
            raise ValueError("No oddball events found in the data.")
        if n_standard == 0:
            raise ValueError("No standard events found in the data.")
        
        # Set random seed for reproducible sampling
        np.random.seed(self.random_seed)
        
        if self.use_fixed_split:
            # Use fixed split: 10+10 train, 5+5 val, 5+5 test
            train_oddball = TRAIN_TRIALS_PER_CLASS
            val_oddball = VAL_TRIALS_PER_CLASS
            test_oddball = TEST_TRIALS_PER_CLASS
            train_standard = TRAIN_TRIALS_PER_CLASS
            val_standard = VAL_TRIALS_PER_CLASS
            test_standard = TEST_TRIALS_PER_CLASS
            
            total_needed_oddball = train_oddball + val_oddball + test_oddball
            total_needed_standard = train_standard + val_standard + test_standard
            
            # Check if we have enough events
            if n_oddball < total_needed_oddball:
                print(f"Warning: Only {n_oddball} oddball events available, need {total_needed_oddball}")
                # Adjust proportions
                train_oddball = min(train_oddball, n_oddball // 3)
                val_oddball = min(val_oddball, (n_oddball - train_oddball) // 2)
                test_oddball = n_oddball - train_oddball - val_oddball
            
            if n_standard < total_needed_standard:
                print(f"Warning: Only {n_standard} standard events available, need {total_needed_standard}")
                # Adjust proportions
                train_standard = min(train_standard, n_standard // 3)
                val_standard = min(val_standard, (n_standard - train_standard) // 2)
                test_standard = n_standard - train_standard - val_standard
            
            # Sample events for each split
            oddball_indices = np.random.choice(n_oddball, size=n_oddball, replace=False)
            standard_indices = np.random.choice(n_standard, size=n_standard, replace=False)
            
            # Split oddball events
            oddball_train = oddball_events[oddball_indices[:train_oddball]]
            oddball_val = oddball_events[oddball_indices[train_oddball:train_oddball+val_oddball]]
            oddball_test = oddball_events[oddball_indices[train_oddball+val_oddball:train_oddball+val_oddball+test_oddball]]
            
            # Split standard events
            standard_train = standard_events[standard_indices[:train_standard]]
            standard_val = standard_events[standard_indices[train_standard:train_standard+val_standard]]
            standard_test = standard_events[standard_indices[train_standard+val_standard:train_standard+val_standard+test_standard]]
            
            # Combine all events and create labels
            all_events = np.vstack([
                oddball_train, standard_train,  # train: 0-19
                oddball_val, standard_val,      # val: 20-29
                oddball_test, standard_test     # test: 30-39
            ])
            
            # Create labels with split information
            train_labels = np.concatenate([
                np.ones(train_oddball, dtype=int),   # oddball = 1
                np.zeros(train_standard, dtype=int)  # standard = 0
            ])
            val_labels = np.concatenate([
                np.ones(val_oddball, dtype=int),     # oddball = 1
                np.zeros(val_standard, dtype=int)    # standard = 0
            ])
            test_labels = np.concatenate([
                np.ones(test_oddball, dtype=int),    # oddball = 1
                np.zeros(test_standard, dtype=int)   # standard = 0
            ])
            
            labels = np.concatenate([train_labels, val_labels, test_labels])
            
            # Create split indices
            train_end = len(train_labels)
            val_end = train_end + len(val_labels)
            test_end = val_end + len(test_labels)
            
            # Store split information
            self.train_indices = np.arange(0, train_end)
            self.val_indices = np.arange(train_end, val_end)
            self.test_indices = np.arange(val_end, test_end)
            
            selected_events = all_events
            
            print(f"Fixed split dataset: Train({train_oddball}+{train_standard}), Val({val_oddball}+{val_standard}), Test({test_oddball}+{test_standard})")
            
        else:
            # Original logic: use fixed number of trials per class
            target_trials = self.fixed_trials_per_class
            
            # Sample oddball events
            if n_oddball >= target_trials:
                oddball_indices = np.random.choice(n_oddball, size=target_trials, replace=False)
                selected_oddball_events = oddball_events[oddball_indices]
            else:
                # Not enough oddball events - use all available
                selected_oddball_events = oddball_events.copy()
                print(f"Warning: Only {n_oddball} oddball events available, using all of them")
            
            # Sample standard events
            if n_standard >= target_trials:
                standard_indices = np.random.choice(n_standard, size=target_trials, replace=False)
                selected_standard_events = standard_events[standard_indices]
            else:
                # Not enough standard events - use all available
                selected_standard_events = standard_events.copy()
                print(f"Warning: Only {n_standard} standard events available, using all of them")
            
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
            print(f"Fixed trials dataset: {n_selected_oddball} oddball, {n_selected_standard} standard events (target: {target_trials} each)")

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
        
        # Cache the processed data if enabled
        if self.use_cache and self.cache is not None:
            raw_file = getattr(raw, 'filenames', ['unknown'])[0] if hasattr(raw, 'filenames') else 'unknown'
            self.cache.cache_data(
                raw_file=raw_file,
                channels=self.eeg_channels,
                trial_start_offset=self.trial_start_offset_samples,
                trial_stop_offset=self.trial_stop_offset_samples,
                low_freq=LOW_FREQ,
                high_freq=HIGH_FREQ,
                windows_data=windows_data,
                windows_labels=windows_labels,
                resample_freq=RESAMPLE_FREQ
            )
        
        # Return custom dataset
        return ManualWindowsDataset(windows_data, windows_labels)
