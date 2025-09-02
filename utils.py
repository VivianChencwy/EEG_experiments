"""
Utility functions for EEG experiments
"""

import os
import numpy as np
import torch
import pandas as pd
from mne.io import read_raw_eeglab, read_raw_brainvision
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
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
        return read_raw_eeglab(file_path, preload=False)
    else: 
        return read_raw_brainvision(file_path, preload=False)


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
        f"Mean Accuracy: {stats['mean']:.3f}",
        f"95% Confidence Interval: [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]",
        f"Best Subject: {stats['best_subject'][0]} ({stats['best_subject'][1]:.3f})",
        f"Worst Subject: {stats['worst_subject'][0]} ({stats['worst_subject'][1]:.3f})",
    ]
    
    # Calculate overall precision, recall, f1 score if prediction details are provided
    if prediction_details:
        total_precision = np.mean([details['precision'] for details in prediction_details.values() if 'precision' in details])
        total_recall = np.mean([details['recall'] for details in prediction_details.values() if 'recall' in details])
        total_f1 = np.mean([details['f1_score'] for details in prediction_details.values() if 'f1_score' in details])
        total_auc = np.mean([details.get('auc', 0.5) for details in prediction_details.values()])
        
        out_lines.extend([
            f"Mean Precision: {total_precision:.3f}",
            f"Mean Recall: {total_recall:.3f}",
            f"Mean F1-Score: {total_f1:.3f}",
            f"Mean AUC: {total_auc:.3f}",
        ])
    
    for line in out_lines:
        print(line)
        if logger is not None:
            logger.info(line)


def run_experiment_with_seed(train_loader, val_loader, test_loader, n_channels, device, 
                           seed, classifier_type, print_model_summary=False, return_details=False):
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
    
    model = create_model(n_channels, is_lda)
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
                       return_indices=False, use_weighted_sampling=True):
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
    use_weighted_sampling : bool, default True
        If True, use weighted random sampling for training set
        
    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader) or
        (train_loader, val_loader, test_loader, train_indices, val_indices, test_indices)
    """
    temp_size = val_size + test_size
    indices = np.arange(len(data))
    
    train_indices, temp_indices, X_train, X_temp, y_train, y_temp = train_test_split(
        indices, data, labels, test_size=temp_size, stratify=labels
    )
    
    test_ratio = test_size / temp_size  
    val_indices, test_indices, X_val, X_test, y_val, y_test = train_test_split(
        temp_indices, X_temp, y_temp, test_size=test_ratio, stratify=y_temp
    )
    
    # Create weighted sampler for training set to handle class imbalance
    train_sampler = None
    if use_weighted_sampling:
        # Calculate class weights
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        class_weights = total_samples / (len(class_counts) * class_counts)
        
        # Assign weights to each sample
        sample_weights = np.array([class_weights[label] for label in y_train])
        sample_weights = torch.from_numpy(sample_weights).double()
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        print(f"Using weighted sampling - Class distribution: {class_counts.tolist()}, "
              f"Class weights: {class_weights.tolist()}")
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), 
        batch_size=batch_size, 
        sampler=train_sampler,
        shuffle=(train_sampler is None)  # Don't shuffle if using sampler
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
            raw.load_data()
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
