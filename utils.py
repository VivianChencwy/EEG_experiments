"""
Utility functions for EEG experiments
"""

import os
import numpy as np
import torch
from mne.io import read_raw_eeglab, read_raw_brainvision
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy import stats

from models import create_model, train_model, evaluate, normalize_data
from config import BATCH_SIZE, TEST_SIZE, VAL_SIZE, MAX_EPOCHS, seeds
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


def calculate_statistics(accuracies):
    """Calculate mean and 95% confidence interval for accuracies.
    
    Parameters
    ----------
    accuracies : dict
        Dictionary mapping subject IDs to accuracy values
        
    Returns
    -------
    dict
        Dictionary containing statistical measures
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


def print_statistics(stats, dataset_name, logger=None):
    """Print and optionally log statistics in a formatted way.
    
    Parameters
    ----------
    stats : dict
        Statistics dictionary from calculate_statistics
    dataset_name : str
        Name of the dataset for display
    logger : logging.Logger, optional
        Logger to write statistics to
    """
    out_lines = [
        f"\n{dataset_name} Statistics:",
        f"Mean Accuracy: {stats['mean']:.3f}",
        f"95% Confidence Interval: [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]",
        f"Best Subject: {stats['best_subject'][0]} ({stats['best_subject'][1]:.3f})",
        f"Worst Subject: {stats['worst_subject'][0]} ({stats['worst_subject'][1]:.3f})",
    ]
    for line in out_lines:
        print(line)
        if logger is not None:
            logger.info(line)


def run_experiment_with_seed(train_loader, val_loader, test_loader, n_channels, device, 
                           seed, classifier_type, print_model_summary=False):
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
        
    Returns
    -------
    tuple
        (accuracy, model) tuple
    """
    is_lda = classifier_type.lower() == 'lda'
    
    if not is_lda:
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        np.random.seed(seed)
    
    model = create_model(n_channels, is_lda)
    if not is_lda:
        model = model.to(device)
        # Print model summary only once per experiment (for the first seed)
        if print_model_summary and seed == seeds[0]:
            print("\n" + "="*60)
            print("ShallowFBCSPNet Model Architecture Summary")
            print("="*60)
            print(f"Model: {model}")
            print(f"Input channels: {n_channels}")
            print(f"Input shape: (batch_size, {n_channels}, 128)")
            print("="*60 + "\n")
    
    accuracy = train_model(model, train_loader, val_loader, test_loader, device, is_lda, MAX_EPOCHS)
    return accuracy, model


def create_data_loaders(data, labels, batch_size=BATCH_SIZE, test_size=TEST_SIZE, val_size=VAL_SIZE):
    """Create train, validation, and test data loaders.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    labels : np.ndarray
        Target labels
    batch_size : int, default from config
        Batch size for data loaders
    test_size : float, default from config
        Proportion of data to use for testing
    val_size : float, default from config
        Proportion of remaining data to use for validation
        
    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader) tuple
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, labels, test_size=test_size, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp
    )
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), 
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), 
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), 
        batch_size=batch_size, shuffle=False
    )
    
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

        # Prepare data
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