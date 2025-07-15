"""
Preprocessor classes for EEG experiments
"""

import numpy as np
import mne
from braindecode.preprocessing import Preprocessor, create_windows_from_events
from braindecode.datasets import BaseConcatDataset, BaseDataset

from constants import RESPONSE_EVENTS, ODDBALL_EVENTS, EVENT_MAPPING
from config import (
    TRIAL_START_OFFSET_SAMPLES, TRIAL_STOP_OFFSET_SAMPLES,
    LOW_FREQ, HIGH_FREQ, RESAMPLE_FREQ
)


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

        # Remove last event
        events = events[:-1]

        # Drop response events
        response_mask = np.isin(events[:, 2], RESPONSE_EVENTS)
        events = events[~response_mask]

        # Map oddball codes → 1, standard → 0
        oddball_mask = np.isin(events[:, 2], ODDBALL_EVENTS)
        new_events = np.zeros_like(events)
        new_events[:, 0] = events[:, 0]
        new_events[oddball_mask, 2] = 1

        # Attach new annotations
        annot_from_events = mne.annotations_from_events(
            events=new_events,
            event_desc=EVENT_MAPPING,
            sfreq=raw.info["sfreq"],
        )
        raw.set_annotations(annot_from_events)

        # Create windowed dataset
        windows_ds = create_windows_from_events(
            BaseConcatDataset([BaseDataset(raw, target_name=None)]),
            trial_start_offset_samples=self.trial_start_offset_samples,
            trial_stop_offset_samples=self.trial_stop_offset_samples,
            preload=False,
        )

        return windows_ds 