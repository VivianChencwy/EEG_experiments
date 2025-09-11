"""
Advanced preprocessor with ICA artifact removal and CSP spatial filtering for enhanced EEG classification
"""

import numpy as np
import mne
from mne.preprocessing import ICA
from braindecode.preprocessing import Preprocessor
from braindecode.datasets import BaseConcatDataset, BaseDataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import signal
import warnings

from constants import RESPONSE_EVENTS, ODDBALL_EVENTS, EVENT_MAPPING
from config import (
    TRIAL_START_OFFSET_SAMPLES, TRIAL_STOP_OFFSET_SAMPLES,
    LOW_FREQ, HIGH_FREQ, RESAMPLE_FREQ
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


class CommonSpatialPatterns:
    """Common Spatial Patterns (CSP) implementation for EEG spatial filtering."""
    
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_ = None
        self.eigenvalues_ = None
        
    def fit(self, X, y):
        """
        Fit CSP filters.
        
        Parameters
        ----------
        X : array-like, shape (n_trials, n_channels, n_samples)
            Training data
        y : array-like, shape (n_trials,)
            Training labels
        """
        X = np.array(X)
        y = np.array(y)
        
        # Separate classes
        class_0_mask = y == 0
        class_1_mask = y == 1
        
        if not np.any(class_0_mask) or not np.any(class_1_mask):
            raise ValueError("Both classes must be present in the data")
        
        X_0 = X[class_0_mask]
        X_1 = X[class_1_mask]
        
        # Compute covariance matrices
        C_0 = np.zeros((X.shape[1], X.shape[1]))
        for trial in X_0:
            C_0 += np.cov(trial)
        C_0 /= len(X_0)
        
        C_1 = np.zeros((X.shape[1], X.shape[1]))
        for trial in X_1:
            C_1 += np.cov(trial)
        C_1 /= len(X_1)
        
        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = np.linalg.eigh(C_1, C_0 + C_1)
        
        # Sort eigenvalues in descending order
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]
        
        # Select components (most discriminative)
        n_select = min(self.n_components // 2, len(eigenvalues) // 2)
        selected_filters = np.column_stack([
            eigenvectors[:, :n_select],  # Largest eigenvalues
            eigenvectors[:, -n_select:]  # Smallest eigenvalues
        ])
        
        self.filters_ = selected_filters.T
        self.eigenvalues_ = eigenvalues
        
        return self
        
    def transform(self, X):
        """
        Apply CSP spatial filtering.
        
        Parameters
        ----------
        X : array-like, shape (n_trials, n_channels, n_samples)
            Data to transform
            
        Returns
        -------
        X_csp : array, shape (n_trials, n_components, n_samples)
            CSP transformed data
        """
        if self.filters_ is None:
            raise ValueError("CSP filters not fitted. Call fit() first.")
        
        X = np.array(X)
        X_csp = np.zeros((X.shape[0], self.filters_.shape[0], X.shape[2]))
        
        for i, trial in enumerate(X):
            X_csp[i] = np.dot(self.filters_, trial)
            
        return X_csp


class AdvancedOddballPreprocessor(Preprocessor):
    """Advanced preprocessor with ICA artifact removal and CSP spatial filtering."""

    def __init__(self, eeg_channels, 
                 trial_start_offset_samples=TRIAL_START_OFFSET_SAMPLES,
                 trial_stop_offset_samples=TRIAL_STOP_OFFSET_SAMPLES,
                 random_seed=42,
                 use_cache=True,
                 use_ica=True,
                 use_csp=True,
                 n_csp_components=8,
                 ica_n_components=15,
                 freq_band=(8, 30)):  # Optimized for ERP
        super().__init__(fn=self.transform, apply_on_array=False)
        self.eeg_channels = [ch.lower() for ch in eeg_channels]
        self.trial_start_offset_samples = trial_start_offset_samples
        self.trial_stop_offset_samples = trial_stop_offset_samples
        self.random_seed = random_seed
        self.use_cache = use_cache
        self.use_ica = use_ica
        self.use_csp = use_csp
        self.n_csp_components = n_csp_components
        self.ica_n_components = ica_n_components
        self.freq_band = freq_band
        self.cache = EEGDataCache() if use_cache else None
        self.csp_filter = None
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)

    def _apply_ica_artifact_removal(self, raw):
        """Apply ICA-based artifact removal."""
        try:
            # Create a copy to avoid modifying original data
            raw_copy = raw.copy()
            
            # High-pass filter for ICA (recommended: 1Hz for mobile/artifact-heavy data)
            raw_for_ica = raw_copy.copy().filter(l_freq=1.0, h_freq=None)
            
            # Fit ICA
            ica = ICA(
                n_components=min(self.ica_n_components, len(raw_for_ica.ch_names)),
                random_state=self.random_seed,
                max_iter="auto"
            )
            
            # Fit ICA on filtered data
            ica.fit(raw_for_ica)
            
            # Identify artifacts using correlation-based approach
            # Find components with high correlation to frontal electrodes (eye artifacts)
            frontal_channels = [ch for ch in ['fp1', 'fp2', 'f7', 'f8'] 
                              if ch in [c.lower() for c in raw_copy.ch_names]]
            
            if frontal_channels:
                # Get component time courses
                ica_sources = ica.get_sources(raw_for_ica).get_data()
                
                # Get frontal electrode data
                frontal_picks = [raw_copy.ch_names.index(ch) for ch in raw_copy.ch_names 
                               if ch.lower() in frontal_channels]
                frontal_data = raw_copy.get_data(picks=frontal_picks)
                frontal_avg = np.mean(frontal_data, axis=0)
                
                # Find components correlated with frontal activity (eye artifacts)
                exclude_components = []
                for comp_idx in range(ica_sources.shape[0]):
                    corr = np.corrcoef(ica_sources[comp_idx], frontal_avg)[0, 1]
                    if abs(corr) > 0.7:  # High correlation threshold
                        exclude_components.append(comp_idx)
                
                # Limit to max 3 components to avoid over-removal
                exclude_components = exclude_components[:3]
                
                if exclude_components:
                    ica.exclude = exclude_components
                    print(f"ICA: Excluding {len(exclude_components)} artifact components")
                
            # Apply ICA to remove artifacts
            raw_clean = raw_copy.copy()
            ica.apply(raw_clean)
            
            return raw_clean
            
        except Exception as e:
            print(f"ICA failed: {e}. Using original data.")
            return raw.copy()

    def _extract_csp_features(self, windows_data, windows_labels):
        """Extract CSP features from windowed data."""
        if not self.use_csp:
            return windows_data
            
        try:
            # Fit CSP on training data
            if self.csp_filter is None:
                self.csp_filter = CommonSpatialPatterns(n_components=self.n_csp_components)
                self.csp_filter.fit(windows_data, windows_labels)
            
            # Transform data
            csp_data = self.csp_filter.transform(windows_data)
            print(f"CSP: Transformed from {windows_data.shape[1]} to {csp_data.shape[1]} channels")
            return csp_data
            
        except Exception as e:
            print(f"CSP failed: {e}. Using original data.")
            return windows_data

    def _apply_advanced_filtering(self, raw):
        """Apply advanced multi-band filtering."""
        try:
            # Apply optimal frequency band for ERP/P300 detection
            raw.filter(l_freq=self.freq_band[0], h_freq=self.freq_band[1], 
                      method='fir', fir_design='firwin')
            
            # Apply notch filter for power line noise
            raw.notch_filter(freqs=[50, 60], method='fir', fir_design='firwin')
            
            return raw
            
        except Exception as e:
            print(f"Advanced filtering failed: {e}. Using basic filtering.")
            raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ)
            return raw

    def transform(self, raw):
        """Transform raw EEG data with advanced preprocessing."""
        # Check cache first if enabled
        if self.use_cache and self.cache is not None:
            raw_file = getattr(raw, 'filenames', ['unknown'])[0] if hasattr(raw, 'filenames') else 'unknown'
            
            cache_key = f"{raw_file}_advanced_{self.use_ica}_{self.use_csp}_{self.freq_band}"
            cached_result = self.cache.get_cached_data(
                raw_file=cache_key,
                channels=self.eeg_channels,
                trial_start_offset=self.trial_start_offset_samples,
                trial_stop_offset=self.trial_stop_offset_samples,
                low_freq=self.freq_band[0],
                high_freq=self.freq_band[1],
                resample_freq=RESAMPLE_FREQ
            )
            
            if cached_result is not None:
                windows_data, windows_labels = cached_result
                return ManualWindowsDataset(windows_data, windows_labels)
        
        # Standardize channel names
        raw.rename_channels({ch: ch.lower() for ch in raw.ch_names})

        # Select available channels
        available_channels = [ch for ch in self.eeg_channels if ch in raw.ch_names]
        if not available_channels:
            raise ValueError(f"None of the requested channels found. Available: {raw.ch_names}")

        raw.pick_channels(available_channels)
        
        # Apply ICA artifact removal
        if self.use_ica:
            print("Applying ICA artifact removal...")
            raw = self._apply_ica_artifact_removal(raw)
        
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
            raw._data *= 1e6
        elif np.std(raw_data_before) == 0:
            raise ValueError("Data is constant or zero")
        
        # Apply advanced filtering
        raw = self._apply_advanced_filtering(raw)
        raw.resample(RESAMPLE_FREQ)

        # Extract events
        events, _ = mne.events_from_annotations(raw)
        if len(events) == 0:
            raise ValueError("No events found after reading annotations.")

        # Remove response events
        response_mask = np.isin(events[:, 2], RESPONSE_EVENTS)
        events = events[~response_mask]
        if len(events) == 0:
            raise ValueError("No non-response events found after filtering.")

        # Remove last event to avoid overflow
        events = events[:-1]
        
        # Balance oddball and standard events
        oddball_mask = np.isin(events[:, 2], ODDBALL_EVENTS)
        oddball_events = events[oddball_mask]
        standard_events = events[~oddball_mask]
        
        if len(oddball_events) == 0 or len(standard_events) == 0:
            raise ValueError("Both oddball and standard events required.")
        
        # Use all oddball events and match with standard events
        selected_oddball_events = oddball_events.copy()
        
        np.random.seed(self.random_seed)
        if len(standard_events) >= len(oddball_events):
            standard_indices = np.random.choice(len(standard_events), 
                                              size=len(oddball_events), replace=False)
            selected_standard_events = standard_events[standard_indices]
        else:
            selected_standard_events = standard_events.copy()
        
        # Combine events and create labels
        selected_events = np.vstack([selected_oddball_events, selected_standard_events])
        labels = np.concatenate([
            np.ones(len(selected_oddball_events), dtype=int),
            np.zeros(len(selected_standard_events), dtype=int)
        ])
        
        print(f"Advanced preprocessing: {len(selected_oddball_events)} oddball, "
              f"{len(selected_standard_events)} standard events")

        # Manual window extraction
        raw_data = raw.get_data()
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
        
        # Apply CSP spatial filtering
        if self.use_csp:
            print("Applying CSP spatial filtering...")
            windows_data = self._extract_csp_features(windows_data, windows_labels)
        
        # Data validation
        if np.any(np.isnan(windows_data)) or np.any(np.isinf(windows_data)):
            print("Warning: Data contains NaN or infinite values, cleaning...")
            windows_data = np.nan_to_num(windows_data, nan=0.0, posinf=1.0, neginf=-1.0)
        
        print(f"Final dataset: {len(windows_data)} windows, {windows_data.shape[1]} channels, "
              f"{windows_data.shape[2]} time points")
        
        # Cache the processed data if enabled
        if self.use_cache and self.cache is not None:
            raw_file = getattr(raw, 'filenames', ['unknown'])[0] if hasattr(raw, 'filenames') else 'unknown'
            cache_key = f"{raw_file}_advanced_{self.use_ica}_{self.use_csp}_{self.freq_band}"
            self.cache.cache_data(
                raw_file=cache_key,
                channels=self.eeg_channels,
                trial_start_offset=self.trial_start_offset_samples,
                trial_stop_offset=self.trial_stop_offset_samples,
                low_freq=self.freq_band[0],
                high_freq=self.freq_band[1],
                windows_data=windows_data,
                windows_labels=windows_labels,
                resample_freq=RESAMPLE_FREQ
            )
        
        return ManualWindowsDataset(windows_data, windows_labels)