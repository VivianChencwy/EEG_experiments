"""
Utility functions for EEG experiments
"""

import os
import numpy as np
import torch
import pandas as pd
from mne.io import read_raw_eeglab, read_raw_brainvision
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy import stats

from models import create_model, train_model, evaluate, normalize_data
from config import BATCH_SIZE, TRAIN_SIZE, VAL_SIZE, TEST_SIZE, MAX_EPOCHS, seeds
from experiment_logger import log_error


def load_raw(file_path, dataset_type):
    """Load raw EEG data based on dataset type.
    
    Parameters
    ----------
    file_path : str
        Path to the EEG file
    dataset_type : str
        Type of dataset ('P3' or 'AVO')
        
    Returns
    -------
    mne.io.Raw
        Raw EEG data object
    """
    if dataset_type == 'P3': 
        return read_raw_eeglab(file_path, preload=True)
    else: 
        return read_raw_brainvision(file_path, preload=True)


def load_events_tsv(subject_id, dataset_dir):
    """Load events from TSV file for a P3 subject.
    
    Parameters
    ----------
    subject_id : str
        Subject ID (e.g., 'sub-001')
    dataset_dir : str
        Path to dataset directory
        
    Returns
    -------
    pd.DataFrame or None
        Events dataframe with columns including 'value', or None if file not found
    """
    try:
        events_file = os.path.join(dataset_dir, subject_id, 'eeg', f'{subject_id}_task-P3_events.tsv')
        if os.path.exists(events_file):
            events_df = pd.read_csv(events_file, sep='\t')
            return events_df
        else:
            print(f"Warning: Events file not found: {events_file}")
            return None
    except Exception as e:
        print(f"Error loading events file: {e}")
        return None


def get_stimulus_event_values(events_df):
    """Extract stimulus event values from events dataframe.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Events dataframe from TSV file
        
    Returns
    -------
    list
        List of stimulus event values in order
    """
    if events_df is None:
        return []
    
    # Filter for stimulus events only (not response events)
    stimulus_events = events_df[events_df['trial_type'] == 'stimulus']
    
    # Extract the 'value' column
    event_values = stimulus_events['value'].tolist()
    
    return event_values


def print_event_values_for_splits(subject_id, dataset_dir, train_indices, val_indices, test_indices):
    """Print event values for train/val/test splits for a specific subject.
    
    Parameters
    ----------
    subject_id : str
        Subject ID (e.g., 'sub-001')
    dataset_dir : str
        Path to dataset directory
    train_indices : array-like
        Indices of training samples
    val_indices : array-like  
        Indices of validation samples
    test_indices : array-like
        Indices of test samples
    """
    # Load events TSV file
    events_df = load_events_tsv(subject_id, dataset_dir)
    if events_df is None:
        print(f"Could not load events for {subject_id}")
        return
    
    # Get stimulus event values
    event_values = get_stimulus_event_values(events_df)
    if not event_values:
        print(f"No stimulus events found for {subject_id}")
        return
    
    # Note: The indices correspond to the processed trials after preprocessing
    # which should match the order of stimulus events in the TSV file
    
    print(f"\nEvent values for {subject_id}:")
    
    # Print train event values
    train_events = [event_values[i] for i in train_indices if i < len(event_values)]
    print(f"Train events ({len(train_events)}): {train_events}")
    
    # Print validation event values
    val_events = [event_values[i] for i in val_indices if i < len(event_values)]
    print(f"Validation events ({len(val_events)}): {val_events}")
    
    # Print test event values
    test_events = [event_values[i] for i in test_indices if i < len(event_values)]
    print(f"Test events ({len(test_events)}): {test_events}")
    
    print()


def calculate_statistics(accuracies):
    """
    Calculate mean and 95% confidence interval for accuracies.
    """
    values = np.array(list(accuracies.values()))
    mean = np.mean(values)
    ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=stats.sem(values))
    best_subject = max(accuracies.items(), key=lambda x: x[1])
    worst_subject = min(accuracies.items(), key=lambda x: x[1])
    
    return {
        'mean': mean,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'best_subject': best_subject,
        'worst_subject': worst_subject
    }


def print_statistics(stats, dataset_name, logger=None, prediction_details=None):
    """
    Print and optionally log statistics in a formatted way.
    """
    out_lines = [
        f"\n{dataset_name} Statistics:",
        f"95% Confidence Interval: [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]",
        f"Best Subject: {stats['best_subject'][0]} ({stats['best_subject'][1]:.3f})",
        f"Worst Subject: {stats['worst_subject'][0]} ({stats['worst_subject'][1]:.3f})",
    ]
    
    # Calculate overall metrics if prediction details are provided
    if prediction_details:
        # Calculate mean confusion matrix metrics
        avg_tp = np.mean([details.get('tp', 0) for details in prediction_details.values()])
        avg_tn = np.mean([details.get('tn', 0) for details in prediction_details.values()])
        avg_fp = np.mean([details.get('fp', 0) for details in prediction_details.values()])
        avg_fn = np.mean([details.get('fn', 0) for details in prediction_details.values()])
        
        # Calculate accuracy from confusion matrix
        total_accuracy = (avg_tp + avg_tn) / (avg_tp + avg_tn + avg_fp + avg_fn) if (avg_tp + avg_tn + avg_fp + avg_fn) > 0 else 0
        
        # Debug: Print confusion matrix calculation
        print(f"DEBUG: Confusion Matrix Calculation:")
        print(f"  Avg TP: {avg_tp:.1f}, Avg TN: {avg_tn:.1f}")
        print(f"  Avg FP: {avg_fp:.1f}, Avg FN: {avg_fn:.1f}")
        print(f"  Total Accuracy: {total_accuracy:.3f}")
        
        # Calculate precision, recall, f1 from confusion matrix metrics
        total_precision = avg_tp / (avg_tp + avg_fp) if (avg_tp + avg_fp) > 0 else 0
        total_recall = avg_tp / (avg_tp + avg_fn) if (avg_tp + avg_fn) > 0 else 0
        total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
        
        # Calculate AUC (using provided values)
        auc_values = [details.get('auc', 0.5) for details in prediction_details.values()]
        valid_auc_values = [auc for auc in auc_values if not np.isnan(auc)]
        total_auc = np.mean(valid_auc_values) if valid_auc_values else 0.5
        
        out_lines.extend([
            f"Mean Accuracy: {total_accuracy:.3f}",
            f"Mean Precision: {total_precision:.3f}",
            f"Mean Recall: {total_recall:.3f}",
            f"Mean F1-Score: {total_f1:.3f}",
            f"Mean AUC: {total_auc:.3f}",
            f"Mean Confusion Matrix:",
            f"  TP: {int(round(avg_tp))}, TN: {int(round(avg_tn))}",
            f"  FP: {int(round(avg_fp))}, FN: {int(round(avg_fn))}"
        ])
    
    for line in out_lines:
        print(line)
        if logger is not None:
            logger.info(line)


def run_experiment_with_seed(train_loader, val_loader, test_loader, n_channels, device,
                           seed, classifier_type, print_model_summary=False, return_details=False, input_channels=None):
    """Run a single experiment with a specific random seed.
    
    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        Training data loader
    val_loader : torch.utils.data.DataLoader
        Validation data loader
    test_loader : torch.utils.data.DataLoader
        Test data loader
    n_channels : int
        Number of input channels
    device : torch.device
        Device to run on
    seed : int
        Random seed
    classifier_type : str
        Type of classifier ('lda' or other)
    print_model_summary : bool, default False
        Whether to print model summary
    return_details : bool, default False
        Whether to return detailed prediction counts
        
    Returns
    -------
    tuple
        (accuracy, model) or (details_dict, model) tuple
    """
    is_lda = classifier_type.lower() == 'lda'
    
    if not is_lda:
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        np.random.seed(seed)
    
    model = create_model(n_channels, is_lda, input_channels=input_channels)
    if not is_lda:
        # Only neural network models need to be moved to device
        if hasattr(model, 'to'):
            model = model.to(device)
        # Print model summary only once per experiment (for the first seed)
        if print_model_summary and seed == seeds[0]:
            print("\n" + "="*60)
            print("ShallowFBCSPNet Model Architecture Summary")
            print("="*60)
            print(f"Model type: {type(model).__name__}")
            print(f"Input channels: {n_channels}")
            print(f"Input shape: (batch_size, {n_channels}, 128)")
            if hasattr(model, 'parameters'):
                print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
            print("="*60 + "\n")
    
    # Train the model
    train_model(model, train_loader, val_loader, test_loader, device, is_lda, MAX_EPOCHS)
    
    # Get test evaluation with details if requested
    if return_details:
        test_result = evaluate(model, test_loader, device, is_lda, return_details=True)
        return test_result, model
    else:
        accuracy = evaluate(model, test_loader, device, is_lda)
        return accuracy, model


def create_data_loaders(data, labels, batch_size=BATCH_SIZE, 
                       train_size=TRAIN_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE,
                       return_indices=False, max_trials_per_split=None):
    """Create train, validation, and test data loaders.
    
    Parameters
    ----------
    data : array-like
        Input data
    labels : array-like
        Target labels  
    batch_size : int, default BATCH_SIZE
        Batch size for data loaders
    train_size : float, default TRAIN_SIZE
        Proportion of data for training
    val_size : float, default VAL_SIZE
        Proportion of data for validation
    test_size : float, default TEST_SIZE
        Proportion of data for testing
    return_indices : bool, default False
        If True, also return the indices for each split
    max_trials_per_split : dict, optional
        Dictionary with keys 'train', 'val', 'test' and values being max trial counts
        e.g., {'train': 100, 'val': 20, 'test': 30}
        
    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader) or
        (train_loader, val_loader, test_loader, train_indices, val_indices, test_indices)
    """
    # Check if we should use fixed trial counts instead of ratios
    if max_trials_per_split is not None and all(max_trials_per_split.get(split) is not None for split in ['train', 'val', 'test']):
        # Use fixed trial counts
        max_train = max_trials_per_split['train']
        max_val = max_trials_per_split['val'] 
        max_test = max_trials_per_split['test']
        
        print(f"Using fixed trial counts: Train={max_train}, Val={max_val}, Test={max_test}")
        
        # Shuffle data first
        indices = np.arange(len(data))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        # Split by fixed counts
        train_indices = indices[:max_train]
        val_indices = indices[max_train:max_train + max_val]
        test_indices = indices[max_train + max_val:max_train + max_val + max_test]
        
        # Extract data
        X_train, y_train = data[train_indices], labels[train_indices]
        X_val, y_val = data[val_indices], labels[val_indices]
        X_test, y_test = data[test_indices], labels[test_indices]
        
    else:
        # Use ratio-based splitting (original logic)
        temp_size = val_size + test_size
        indices = np.arange(len(data))
        
        train_indices, temp_indices, X_train, X_temp, y_train, y_temp = train_test_split(
            indices, data, labels, test_size=temp_size, stratify=labels
        )
        
        test_ratio = test_size / temp_size  
        val_indices, test_indices, X_val, X_test, y_val, y_test = train_test_split(
            temp_indices, X_temp, y_temp, test_size=test_ratio, stratify=y_temp
        )
    
    # Apply trial limits if specified, maintaining class balance (only for ratio-based splitting)
    if max_trials_per_split is not None and not all(max_trials_per_split.get(split) is not None for split in ['train', 'val', 'test']):
        if 'train' in max_trials_per_split and max_trials_per_split['train'] is not None:
            max_train = max_trials_per_split['train']
            if len(X_train) > max_train:
                # Sample while maintaining class balance
                X_train, y_train, train_indices = _balanced_sample(
                    X_train, y_train, train_indices, max_train, seed=42
                )
        
        if 'val' in max_trials_per_split and max_trials_per_split['val'] is not None:
            max_val = max_trials_per_split['val']
            if len(X_val) > max_val:
                # Sample while maintaining class balance
                X_val, y_val, val_indices = _balanced_sample(
                    X_val, y_val, val_indices, max_val, seed=42
                )
        
        if 'test' in max_trials_per_split and max_trials_per_split['test'] is not None:
            max_test = max_trials_per_split['test']
            if len(X_test) > max_test:
                # Sample while maintaining class balance
                X_test, y_test, test_indices = _balanced_sample(
                    X_test, y_test, test_indices, max_test, seed=42
                )
    
    # Debug: Print final class distributions
    print(f"DEBUG: Final class distributions:")
    print(f"  Train: {np.bincount(y_train).tolist()}")
    print(f"  Val:   {np.bincount(y_val).tolist()}")
    print(f"  Test:  {np.bincount(y_test).tolist()}")
    
    # Since dataset is now balanced at source, no need for weighted sampling
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), 
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), 
        batch_size=batch_size, shuffle=False
    )
    
    if return_indices:
        return train_loader, val_loader, test_loader, train_indices, val_indices, test_indices
    else:
        return train_loader, val_loader, test_loader


def _balanced_sample(X, y, indices, max_samples, seed=42):
    """
    Sample data while maintaining class balance (1:1 ratio).
    
    Parameters
    ----------
    X : array-like
        Input data
    y : array-like
        Target labels
    indices : array-like
        Original indices
    max_samples : int
        Maximum number of samples to return
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        (sampled_X, sampled_y, sampled_indices)
    """
    np.random.seed(seed)
    
    # Get unique classes
    unique_classes = np.unique(y)
    if len(unique_classes) != 2:
        print(f"Warning: Expected 2 classes, found {len(unique_classes)}. Using random sampling.")
        if len(X) > max_samples:
            sample_indices = np.random.choice(len(X), max_samples, replace=False)
            return X[sample_indices], y[sample_indices], indices[sample_indices]
        return X, y, indices
    
    # Calculate samples per class (ensure even number for 1:1 ratio)
    samples_per_class = max_samples // 2
    
    # Get indices for each class
    class_0_indices = np.where(y == unique_classes[0])[0]
    class_1_indices = np.where(y == unique_classes[1])[0]
    
    # Check if we have enough samples for each class
    if len(class_0_indices) < samples_per_class or len(class_1_indices) < samples_per_class:
        print(f"Warning: Not enough samples for balanced sampling. Class 0: {len(class_0_indices)}, Class 1: {len(class_1_indices)}, Need: {samples_per_class} each")
        # Use all available samples if not enough for balanced sampling
        if len(X) > max_samples:
            sample_indices = np.random.choice(len(X), max_samples, replace=False)
            return X[sample_indices], y[sample_indices], indices[sample_indices]
        return X, y, indices
    
    # Sample from each class
    class_0_sample = np.random.choice(class_0_indices, samples_per_class, replace=False)
    class_1_sample = np.random.choice(class_1_indices, samples_per_class, replace=False)
    
    # Combine samples
    sample_indices = np.concatenate([class_0_sample, class_1_sample])
    np.random.shuffle(sample_indices)  # Shuffle to mix classes
    
    # Debug: Verify class balance
    sampled_y = y[sample_indices]
    class_counts = np.bincount(sampled_y)
    print(f"DEBUG: Balanced sampling - Class distribution: {class_counts.tolist()}")
    
    return X[sample_indices], y[sample_indices], indices[sample_indices]


def get_trial_limits_from_config():
    """Get trial limits from configuration.
    
    Returns
    -------
    dict or None
        Dictionary with trial limits or None if no limits are set
    """
    from config import (
        MAX_TRIALS_PER_SUBJECT_TRAIN, MAX_TRIALS_PER_SUBJECT_VAL, MAX_TRIALS_PER_SUBJECT_TEST,
        FIXED_TRIALS_PER_SUBJECT_TRAIN, FIXED_TRIALS_PER_SUBJECT_VAL, FIXED_TRIALS_PER_SUBJECT_TEST
    )
    
    # Check if fixed trial counts are specified (takes priority)
    if any(x is not None for x in [FIXED_TRIALS_PER_SUBJECT_TRAIN, FIXED_TRIALS_PER_SUBJECT_VAL, FIXED_TRIALS_PER_SUBJECT_TEST]):
        return {
            'train': FIXED_TRIALS_PER_SUBJECT_TRAIN,
            'val': FIXED_TRIALS_PER_SUBJECT_VAL,
            'test': FIXED_TRIALS_PER_SUBJECT_TEST
        }
    
    # Check if max trial limits are specified
    if any(x is not None for x in [MAX_TRIALS_PER_SUBJECT_TRAIN, MAX_TRIALS_PER_SUBJECT_VAL, MAX_TRIALS_PER_SUBJECT_TEST]):
        return {
            'train': MAX_TRIALS_PER_SUBJECT_TRAIN,
            'val': MAX_TRIALS_PER_SUBJECT_VAL,
            'test': MAX_TRIALS_PER_SUBJECT_TEST
        }
    
    return None


def get_channel_list(electrode_list, dataset_type):
    """Get the appropriate channel list based on configuration.
    
    Parameters
    ----------
    electrode_list : str
        Type of electrode list ('common' or 'all')
    dataset_type : str
        Type of dataset ('P3' or 'AVO')
        
    Returns
    -------
    list
        List of channel names
    """
    from constants import COMMON_CHANNELS, P3_CHANNELS, AVO_CHANNELS
    
    if electrode_list == 'common':
        return COMMON_CHANNELS
    else:
        if dataset_type == 'P3':
            return P3_CHANNELS
        else:
            return AVO_CHANNELS


def process_subject_data(subject_id_or_dir, dataset_dir_or_obj, preprocessor, logger, dataset_type='P3'):
    """Process a single subject's data for either P3 or Active Visual Oddball dataset.
    
    Parameters
    ----------
    subject_id_or_dir : str
        Subject directory name (P3) or subject ID (AVO)
    dataset_dir_or_obj : str or object
        Dataset directory (P3) or dataset object (AVO)
    preprocessor : OddballPreprocessor
        Preprocessor instance
    logger : logging.Logger
        Logger for error reporting
    dataset_type : str, default 'P3'
        Type of dataset ('P3' or 'AVO')
        
    Returns
    -------
    tuple
        (data, labels) tuple or (None, None) if processing failed
    """
    try:
        if dataset_type == 'P3':
            eeg_file = os.path.join(
                dataset_dir_or_obj, subject_id_or_dir, 'eeg', 
                f'{subject_id_or_dir}_task-P3_eeg.set'
            )
            raw = load_raw(eeg_file, dataset_type)
            
            # Basic data validation
            raw_data_loaded = raw.get_data()
            if np.all(raw_data_loaded == 0) or np.std(raw_data_loaded) < 1e-10:
                raise ValueError(f"Invalid data for {subject_id_or_dir}: data is constant or zero")
        elif dataset_type == 'AVO':
            import mne
            all_files = [str(f) for f in dataset_dir_or_obj.get_files()]
            # Only include Visual Oddball (VO) runs
            vhdr_files = [
                f for f in all_files
                if f"sub-{subject_id_or_dir}" in f and 'visualoddball' in f and f.endswith('.vhdr')
            ]
            if not vhdr_files:
                return None, None
            
            # Concatenate all runs/files for the subject
            raws = [load_raw(f, dataset_type) for f in vhdr_files]
            for raw_obj in raws:
                raw_obj.load_data()
            raw = mne.concatenate_raws(raws) if len(raws) > 1 else raws[0]
        else:
            raise ValueError("Unknown dataset_type: must be 'P3' or 'AVO'")

        # Process data
        windows = preprocessor.transform(raw)

        # Handle our custom ManualWindowsDataset
        if hasattr(windows, 'data') and hasattr(windows, 'labels'):
            # Custom dataset - direct access to data and labels
            data = windows.data
            labels = windows.labels
        else:
            # Original braindecode dataset - use indexing
            data = np.stack([windows[i][0] for i in range(len(windows))])
            labels = np.array([windows[i][1] for i in range(len(windows))])
            if labels.ndim > 1:
                labels = np.argmax(labels, axis=1)
            labels = labels.squeeze()

        return data, labels

    except Exception as e:
        if dataset_type == 'P3':
            log_error(logger, "P3", subject_id_or_dir, e)
        else:
            log_error(logger, "Active Visual Oddball", f"sub-{subject_id_or_dir}", e)
        return None, None
