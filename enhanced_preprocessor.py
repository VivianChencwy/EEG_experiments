"""
Enhanced preprocessor for EEG data with artifact removal and feature enhancement
"""

import numpy as np
import mne
from scipy import signal
from sklearn.decomposition import FastICA
import warnings
from pathlib import Path
from preprocessor import OddballPreprocessor, ManualWindowsDataset
from enhanced_cache import EnhancedEEGCache
from config import (
    TRIAL_START_OFFSET_SAMPLES, TRIAL_STOP_OFFSET_SAMPLES,
    LOW_FREQ, HIGH_FREQ, RESAMPLE_FREQ
)


class EnhancedOddballPreprocessor(OddballPreprocessor):
    """Enhanced preprocessor with artifact removal and feature extraction."""

    def __init__(self, eeg_channels,
                 trial_start_offset_samples=TRIAL_START_OFFSET_SAMPLES,
                 trial_stop_offset_samples=TRIAL_STOP_OFFSET_SAMPLES,
                 random_seed=42,
                 use_cache=True,
                 dataset_type='P3',
                 # New parameters for enhanced preprocessing
                 remove_artifacts=True,
                 baseline_correct=True,
                 extract_frequency_features=True,
                 apply_notch_filter=True,
                 notch_freqs=[50, 60],  # Power line frequencies
                 alpha_band=(8, 13),
                 beta_band=(13, 30),
                 theta_band=(4, 8),
                 delta_band=(0.5, 4)):

        super().__init__(eeg_channels, trial_start_offset_samples, trial_stop_offset_samples,
                        random_seed, use_cache, dataset_type)

        # Enhanced preprocessing options
        self.remove_artifacts = remove_artifacts
        self.baseline_correct = baseline_correct
        self.extract_frequency_features = extract_frequency_features
        self.apply_notch_filter = apply_notch_filter
        self.notch_freqs = notch_freqs

        # Frequency bands for feature extraction
        self.alpha_band = alpha_band
        self.beta_band = beta_band
        self.theta_band = theta_band
        self.delta_band = delta_band

        # Use enhanced cache instead of basic cache
        if use_cache:
            self.enhanced_cache = EnhancedEEGCache()
        else:
            self.enhanced_cache = None

    def remove_eye_artifacts_ica(self, raw):
        """Remove eye movement artifacts using Independent Component Analysis."""
        if not self.remove_artifacts:
            return raw

        try:
            print("Applying ICA for artifact removal...")

            # Create a copy to avoid modifying original
            raw_ica = raw.copy()

            # High-pass filter for ICA (recommended: 1 Hz)
            raw_ica.filter(l_freq=1.0, h_freq=None)

            # Set up ICA
            n_components = min(15, len(raw_ica.ch_names))  # Limit components for stability
            ica = mne.preprocessing.ICA(n_components=n_components,
                                      random_state=self.random_seed,
                                      method='fastica',
                                      max_iter=200)

            # Fit ICA
            ica.fit(raw_ica)

            # Automatically detect eye movement components
            # This uses correlation with frontal electrodes
            frontal_channels = []
            for ch in raw.ch_names:
                ch_lower = ch.lower()
                if any(front in ch_lower for front in ['fp1', 'fp2', 'af3', 'af4', 'f3', 'f4']):
                    frontal_channels.append(ch)

            if frontal_channels:
                # Find components correlated with eye movements
                eog_indices, eog_scores = ica.find_bads_eog(raw_ica, ch_name=frontal_channels)
                if eog_indices:
                    print(f"Found {len(eog_indices)} eye movement components: {eog_indices}")
                    ica.exclude = eog_indices

                    # Apply ICA to remove artifacts
                    raw_clean = ica.apply(raw.copy())
                    return raw_clean

            print("No frontal channels found for automatic EOG detection, skipping ICA")
            return raw

        except Exception as e:
            print(f"ICA artifact removal failed: {e}")
            return raw

    def apply_enhanced_filtering(self, raw):
        """Apply enhanced filtering including notch filters."""
        # Apply notch filter to remove power line interference
        if self.apply_notch_filter:
            for freq in self.notch_freqs:
                try:
                    raw.notch_filter(freq, verbose=False)
                    print(f"Applied notch filter at {freq} Hz")
                except:
                    pass

        # Apply bandpass filter with better transition bands
        raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ,
                  fir_design='firwin', verbose=False)

        return raw

    def apply_baseline_correction(self, windows_data):
        """Apply baseline correction to each window."""
        if not self.baseline_correct:
            return windows_data

        # Use pre-stimulus period as baseline (assuming it exists)
        baseline_end_idx = abs(self.trial_start_offset_samples) if self.trial_start_offset_samples < 0 else 10

        if baseline_end_idx > 0 and baseline_end_idx < windows_data.shape[2]:
            # Calculate baseline mean for each channel and trial
            baseline_mean = np.mean(windows_data[:, :, :baseline_end_idx], axis=2, keepdims=True)
            windows_data = windows_data - baseline_mean
            print(f"Applied baseline correction using first {baseline_end_idx} samples")

        return windows_data

    def extract_spectral_features(self, windows_data, sfreq):
        """Extract frequency domain features for each window."""
        if not self.extract_frequency_features:
            return windows_data

        print("Extracting frequency domain features...")

        n_windows, n_channels, n_samples = windows_data.shape

        # Calculate power spectral density for each window
        freqs, psd = signal.welch(windows_data, fs=sfreq, nperseg=min(64, n_samples//2))

        # Extract power in different frequency bands
        frequency_features = []

        for band_name, (low_freq, high_freq) in [
            ('delta', self.delta_band),
            ('theta', self.theta_band),
            ('alpha', self.alpha_band),
            ('beta', self.beta_band)
        ]:
            # Find frequency indices for this band
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(band_mask):
                # Calculate mean power in this band for each channel and window
                band_power = np.mean(psd[:, :, band_mask], axis=2)  # (n_windows, n_channels)
                frequency_features.append(band_power)

        if frequency_features:
            # Stack frequency features as additional channels
            freq_features = np.stack(frequency_features, axis=2)  # (n_windows, n_channels, n_bands)

            # Filter out constant frequency features
            valid_freq_features = []
            for band_idx in range(freq_features.shape[2]):
                band_data = freq_features[:, :, band_idx]
                if np.std(band_data) > 1e-6:  # Has variance
                    valid_freq_features.append(band_data)

            if valid_freq_features:
                freq_features_filtered = np.stack(valid_freq_features, axis=2)
                freq_features_filtered = freq_features_filtered.transpose(0, 2, 1)  # (n_windows, n_bands, n_channels)

                # Reshape to match original data format with slight time variation
                n_features = freq_features_filtered.shape[1] * freq_features_filtered.shape[2]
                freq_base = freq_features_filtered.reshape(n_windows, n_features, 1)

                # Add slight time-based modulation to avoid perfectly constant channels
                time_mod = 1 + 0.005 * np.cos(2 * np.pi * np.linspace(0, 1, n_samples))
                freq_features_expanded = freq_base * time_mod.reshape(1, 1, -1)

                # Concatenate with original time domain data
                enhanced_data = np.concatenate([windows_data, freq_features_expanded], axis=1)
                print(f"Added {n_features} frequency features (filtered), new shape: {enhanced_data.shape}")
                return enhanced_data
            else:
                print("No frequency features added (all were constant)")

        return windows_data

    def extract_time_domain_features(self, windows_data):
        """Extract additional time domain features with variance filtering."""
        n_windows, n_channels, n_samples = windows_data.shape
        features = []

        # Mean absolute value (usually has good variance)
        mav = np.mean(np.abs(windows_data), axis=2, keepdims=True)
        if np.std(mav) > 1e-6:  # Only add if has variance
            features.append(mav)

        # Root mean square (usually has good variance)
        rms = np.sqrt(np.mean(windows_data**2, axis=2, keepdims=True))
        if np.std(rms) > 1e-6:
            features.append(rms)

        # Standard deviation (can be constant for preprocessed data)
        std_vals = np.std(windows_data, axis=2, keepdims=True)
        if np.std(std_vals) > 1e-6:  # Only add if varies across windows/channels
            features.append(std_vals)

        # Zero crossings rate (often varies well)
        zero_crossings = np.sum(np.diff(np.sign(windows_data), axis=2) != 0, axis=2, keepdims=True)
        zcr = zero_crossings / windows_data.shape[2]
        if np.std(zcr) > 1e-6:
            features.append(zcr)

        if features:
            # Concatenate all features and expand to match time dimension
            time_features = np.concatenate(features, axis=1)  # (n_windows, n_features, 1)

            # Instead of repeating across time, create a more structured approach
            # Repeat features but add slight time-based variation to avoid constant channels
            time_grid = np.linspace(0, 1, n_samples).reshape(1, 1, -1)
            time_features_expanded = time_features * (1 + 0.01 * time_grid)  # Small time variation

            # Add to original data
            enhanced_data = np.concatenate([windows_data, time_features_expanded], axis=1)
            print(f"Added {time_features.shape[1]} time domain features (filtered for variance)")
        else:
            print("No time domain features added (all were constant)")
            enhanced_data = windows_data

        return enhanced_data

    def check_data_quality(self, windows_data, windows_labels):
        """Check and report data quality issues."""
        print("\n" + "="*40)
        print("Data Quality Report")
        print("="*40)

        n_windows, n_channels, n_samples = windows_data.shape

        # Check for constant channels
        constant_channels = 0
        near_constant_channels = 0

        for ch in range(n_channels):
            ch_data = windows_data[:, ch, :]
            ch_std = np.std(ch_data)

            if ch_std == 0:
                constant_channels += 1
            elif ch_std < 1e-6:
                near_constant_channels += 1

        print(f"Total channels: {n_channels}")
        print(f"Constant channels: {constant_channels}")
        print(f"Near-constant channels: {near_constant_channels}")
        print(f"Variable channels: {n_channels - constant_channels - near_constant_channels}")

        # Check for NaN or inf values
        nan_count = np.sum(np.isnan(windows_data))
        inf_count = np.sum(np.isinf(windows_data))

        print(f"NaN values: {nan_count}")
        print(f"Infinite values: {inf_count}")

        # Check data range
        data_min = np.min(windows_data)
        data_max = np.max(windows_data)
        data_mean = np.mean(windows_data)
        data_std = np.std(windows_data)

        print(f"Data range: [{data_min:.6f}, {data_max:.6f}]")
        print(f"Data mean: {data_mean:.6f}")
        print(f"Data std: {data_std:.6f}")

        # Check class balance
        unique_labels, counts = np.unique(windows_labels, return_counts=True)
        print(f"Class distribution: {dict(zip(unique_labels, counts))}")

        print("="*40)

        return {
            'constant_channels': constant_channels,
            'near_constant_channels': near_constant_channels,
            'variable_channels': n_channels - constant_channels - near_constant_channels,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'data_range': (data_min, data_max),
            'data_stats': (data_mean, data_std),
            'class_distribution': dict(zip(unique_labels, counts))
        }

    def transform(self, raw):
        """Enhanced transform with artifact removal and feature extraction."""
        print("Starting enhanced preprocessing...")

        # Check enhanced cache first
        if self.use_cache and self.enhanced_cache is not None:
            # Try to get raw file path from the raw object
            raw_file = getattr(raw, 'filenames', ['unknown'])[0] if hasattr(raw, 'filenames') else 'unknown'

            print(f"Checking cache for: {Path(raw_file).name}")

            # Prepare enhancement parameters for cache key
            enhancement_params = {
                'remove_artifacts': self.remove_artifacts,
                'baseline_correct': self.baseline_correct,
                'extract_frequency_features': self.extract_frequency_features,
                'apply_notch_filter': self.apply_notch_filter,
                'notch_freqs': self.notch_freqs,
                'frequency_bands': {
                    'alpha': self.alpha_band,
                    'beta': self.beta_band,
                    'theta': self.theta_band,
                    'delta': self.delta_band
                }
            }

            # Try to get cached data
            cached_result = self.enhanced_cache.get_cached_data(
                raw_file=raw_file,
                channels=self.eeg_channels,
                trial_start_offset=self.trial_start_offset_samples,
                trial_stop_offset=self.trial_stop_offset_samples,
                low_freq=LOW_FREQ,
                high_freq=HIGH_FREQ,
                resample_freq=RESAMPLE_FREQ,
                **enhancement_params
            )

            if cached_result is not None:
                windows_data, windows_labels = cached_result
                print("✓ Using cached enhanced preprocessing data")
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

        # Set reference to average
        try:
            raw.set_eeg_reference('average', projection=True)
        except Exception:
            try:
                if 'cz' in [ch.lower() for ch in raw.ch_names]:
                    raw.set_eeg_reference(['Cz'])
            except Exception:
                pass

        # Check and convert data units if needed
        raw_data_before = raw.get_data()
        if np.std(raw_data_before) < 1e-6 and np.std(raw_data_before) > 0:
            raw._data *= 1e6  # Convert V to μV
        elif np.std(raw_data_before) == 0:
            raise ValueError("Data is constant or zero")

        # Enhanced preprocessing steps
        print("Applying enhanced filtering...")
        raw = self.apply_enhanced_filtering(raw)

        print("Removing artifacts...")
        raw = self.remove_eye_artifacts_ica(raw)

        # Resample
        raw.resample(RESAMPLE_FREQ)

        # Extract events (same as parent class)
        events, _ = mne.events_from_annotations(raw)
        if len(events) == 0:
            raise ValueError("No events found after reading annotations.")

        # Process events same as parent class
        response_mask = np.isin(events[:, 2], self.response_events)
        events = events[~response_mask]
        if len(events) == 0:
            raise ValueError("No non-response events found after filtering.")

        events = events[:-1]

        # Balance dataset
        oddball_mask = np.isin(events[:, 2], self.oddball_events)
        oddball_events = events[oddball_mask]
        standard_events = events[~oddball_mask]

        if len(oddball_events) == 0 or len(standard_events) == 0:
            raise ValueError("Need both oddball and standard events")

        selected_oddball_events = oddball_events.copy()

        np.random.seed(self.random_seed)

        n_oddball = len(oddball_events)
        n_standard = len(standard_events)

        if n_standard >= n_oddball:
            standard_indices = np.random.choice(n_standard, size=n_oddball, replace=False)
            selected_standard_events = standard_events[standard_indices]
        else:
            selected_standard_events = standard_events.copy()

        selected_events = np.vstack([selected_oddball_events, selected_standard_events])

        n_selected_oddball = len(selected_oddball_events)
        n_selected_standard = len(selected_standard_events)
        labels = np.concatenate([
            np.ones(n_selected_oddball, dtype=int),
            np.zeros(n_selected_standard, dtype=int)
        ])

        print(f"Balanced dataset: {n_selected_oddball} oddball, {n_selected_standard} standard events")

        # Manual window extraction
        raw_data = raw.get_data()
        sfreq = raw.info['sfreq']

        windows_data = []
        windows_labels = []

        for i, (event_sample, _, _) in enumerate(selected_events):
            start_sample = event_sample + self.trial_start_offset_samples
            end_sample = event_sample + self.trial_stop_offset_samples

            if start_sample >= 0 and end_sample <= raw_data.shape[1]:
                window_data = raw_data[:, start_sample:end_sample]
                windows_data.append(window_data)
                windows_labels.append(labels[i])

        windows_data = np.array(windows_data)
        windows_labels = np.array(windows_labels)

        # Enhanced processing steps
        print("Applying baseline correction...")
        windows_data = self.apply_baseline_correction(windows_data)

        print("Extracting spectral features...")
        windows_data = self.extract_spectral_features(windows_data, sfreq)

        print("Extracting time domain features...")
        windows_data = self.extract_time_domain_features(windows_data)

        # Data validation and cleaning
        if np.any(np.isnan(windows_data)) or np.any(np.isinf(windows_data)):
            print("Warning: Data contains NaN or infinite values, cleaning...")
            windows_data = np.nan_to_num(windows_data, nan=0.0, posinf=1.0, neginf=-1.0)

        print(f"Enhanced preprocessing complete. Final data shape: {windows_data.shape}")
        print(f"Extracted {len(windows_data)} windows ({np.sum(windows_labels)} oddball, {len(windows_data)-np.sum(windows_labels)} standard)")

        # Data quality check
        quality_report = self.check_data_quality(windows_data, windows_labels)

        # Warn if too many constant channels
        if quality_report['constant_channels'] + quality_report['near_constant_channels'] > windows_data.shape[1] * 0.3:
            print(f"WARNING: {quality_report['constant_channels'] + quality_report['near_constant_channels']} out of {windows_data.shape[1]} channels have very low variance!")

        # Cache the enhanced preprocessing results
        if self.use_cache and self.enhanced_cache is not None:
            raw_file = getattr(raw, 'filenames', ['unknown'])[0] if hasattr(raw, 'filenames') else 'unknown'

            # Prepare the same enhancement parameters used for cache lookup
            enhancement_params = {
                'remove_artifacts': self.remove_artifacts,
                'baseline_correct': self.baseline_correct,
                'extract_frequency_features': self.extract_frequency_features,
                'apply_notch_filter': self.apply_notch_filter,
                'notch_freqs': self.notch_freqs,
                'frequency_bands': {
                    'alpha': self.alpha_band,
                    'beta': self.beta_band,
                    'theta': self.theta_band,
                    'delta': self.delta_band
                }
            }

            print("Saving to enhanced cache...")
            self.enhanced_cache.cache_data(
                raw_file=raw_file,
                channels=self.eeg_channels,
                trial_start_offset=self.trial_start_offset_samples,
                trial_stop_offset=self.trial_stop_offset_samples,
                low_freq=LOW_FREQ,
                high_freq=HIGH_FREQ,
                resample_freq=RESAMPLE_FREQ,
                windows_data=windows_data,
                windows_labels=windows_labels,
                **enhancement_params
            )

        return ManualWindowsDataset(windows_data, windows_labels)