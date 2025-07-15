"""
Experiment logic for EEG experiments
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from eegdash.data_utils import EEGBIDSDataset

from config import (
    P3_DATA_DIR, AVO_DATA_DIR, BATCH_SIZE, seeds, 
    use_combined_datasets, separate_subject_classification, 
    electrode_list, classifier
)
from constants import COMMON_CHANNELS, P3_CHANNELS, AVO_CHANNELS
from preprocessor import OddballPreprocessor
from models import create_model, train_model, evaluate, normalize_data
from utils import (
    get_channel_list, process_subject_data, create_data_loaders,
    run_experiment_with_seed, calculate_statistics, print_statistics
)
from experiment_logger import log_individual_results, log_section_header


def train_combined_model(p3_dir, avo_dataset, channels, logger):
    """Train a combined model using both P3 and AVO datasets.
    
    Parameters
    ----------
    p3_dir : str
        Path to P3 dataset directory
    avo_dataset : EEGBIDSDataset
        AVO dataset object
    channels : list
        List of channel names to use
    logger : logging.Logger
        Logger for experiment tracking
        
    Returns
    -------
    dict
        Dictionary of final accuracies per subject
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_data = []
    all_labels = []
    subject_ranges = []
    subject_ids = []
    start_idx = 0
    
    # Process P3 subjects
    p3_preprocessor = OddballPreprocessor(channels)
    for subject_dir in sorted(os.listdir(p3_dir)):
        if not subject_dir.startswith('sub-'):
            continue
        print(f"Loading P3 subject {subject_dir} ...", flush=True)
        data, labels = process_subject_data(subject_dir, p3_dir, p3_preprocessor, logger, dataset_type='P3')
        if data is not None and labels is not None:
            if labels.ndim > 1:
                labels = np.argmax(labels, axis=1)
            labels = labels.squeeze()
            all_data.append(data)
            all_labels.append(labels)
            end_idx = start_idx + len(data)
            subject_ranges.append((start_idx, end_idx))
            subject_ids.append(f"P3_{subject_dir}")
            start_idx = end_idx
    
    # Process AVO subjects
    avo_preprocessor = OddballPreprocessor(channels)
    all_files = [str(f) for f in avo_dataset.get_files()]
    all_subjects = sorted(list(set([f.split('sub-')[1][:3] for f in all_files if 'sub-' in f])))
    
    for subject_id in all_subjects:
        print(f"Loading AVO subject sub-{subject_id} ...", flush=True)
        data, labels = process_subject_data(subject_id, avo_dataset, avo_preprocessor, logger, dataset_type='AVO')
        if data is not None and labels is not None:
            if labels.ndim > 1:
                labels = np.argmax(labels, axis=1)
            labels = labels.squeeze()
            all_data.append(data)
            all_labels.append(labels)
            end_idx = start_idx + len(data)
            subject_ranges.append((start_idx, end_idx))
            subject_ids.append(f"AVO_sub-{subject_id}")
            start_idx = end_idx
    
    if not all_data:
        logger.error("No data available for combined model training")
        return {}
    
    # Combine all data
    all_data = np.concatenate(all_data)
    all_labels = np.concatenate(all_labels)
    
    # Create splits
    train_indices, temp_indices = train_test_split(
        range(len(all_data)), test_size=0.4, stratify=all_labels
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, stratify=all_labels[temp_indices]
    )
    
    # Ensure indices are numpy arrays for vectorized comparisons
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(all_data[train_indices]),
            torch.LongTensor(all_labels[train_indices])
        ),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(all_data[val_indices]),
            torch.LongTensor(all_labels[val_indices])
        ),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(all_data[test_indices]),
            torch.LongTensor(all_labels[test_indices])
        ),
        batch_size=BATCH_SIZE, shuffle=False
    )
    
    # Train models with different seeds
    model_accuracies = {}
    for seed in seeds:
        print(f"Training combined model with seed {seed} ...", flush=True)
        is_lda = classifier.lower() == 'lda'

        if is_lda:
            # Train LDA with seed control
            np.random.seed(seed)
            X_train = []
            y_train = []
            for batch_X, batch_y in train_loader:
                X_train.append(batch_X.reshape(batch_X.shape[0], -1).numpy())
                y_train.append(batch_y.numpy())
            X_train = np.concatenate(X_train)
            y_train = np.concatenate(y_train)
            lda_model = create_model(len(channels), is_lda=True)
            lda_model.fit(X_train, y_train)
            model = lda_model
        else:
            # Train neural network
            torch.manual_seed(seed)
            np.random.seed(seed)
            nn_model = create_model(len(channels), is_lda=False)
            nn_model = nn_model.to(device)
            
            # Print model summary only once (for the first seed)
            if seed == seeds[0]:
                print("\n" + "="*60)
                print("ShallowFBCSPNet Model Architecture Summary (Combined Dataset)")
                print("="*60)
                print(f"Model: {nn_model}")
                print(f"Input channels: {len(channels)}")
                print(f"Input shape: (batch_size, {len(channels)}, 128)")
                print("="*60 + "\n")
            
            train_model(nn_model, train_loader, val_loader, test_loader, device, is_lda=False)
            model = nn_model

        # Evaluate per subject
        subject_accuracies = {}
        if not is_lda:
            with torch.no_grad():
                for subject_idx, (start_idx, end_idx) in enumerate(subject_ranges):
                    mask = (test_indices >= start_idx) & (test_indices < end_idx)
                    subject_test_indices = test_indices[mask]
                    if len(subject_test_indices) > 0:
                        subject_data = torch.FloatTensor(all_data[subject_test_indices])
                        subject_labels = torch.LongTensor(all_labels[subject_test_indices])
                        subject_loader = DataLoader(
                            TensorDataset(subject_data, subject_labels),
                            batch_size=BATCH_SIZE, shuffle=False
                        )
                        subject_acc = evaluate(model, subject_loader, device)
                        subject_accuracies[subject_ids[subject_idx]] = subject_acc
        else:
            for subject_idx, (start_idx, end_idx) in enumerate(subject_ranges):
                mask = (test_indices >= start_idx) & (test_indices < end_idx)
                subject_test_indices = test_indices[mask]
                if len(subject_test_indices) > 0:
                    subject_data = all_data[subject_test_indices]
                    subject_labels = all_labels[subject_test_indices]
                    X = subject_data.reshape(subject_data.shape[0], -1)
                    y = subject_labels
                    lda_predictions = model.predict(X)
                    subject_acc = np.mean(lda_predictions == y)
                    subject_accuracies[subject_ids[subject_idx]] = subject_acc
        
        model_accuracies[f"seed_{seed}"] = subject_accuracies
    
    # Average across seeds
    final_accuracies = {}
    for subject_id in subject_ids:
        accuracies = [model_accuracies[f"seed_{seed}"][subject_id] for seed in seeds]
        final_accuracies[subject_id] = np.mean(accuracies)
    
    return final_accuracies


def train_single_dataset_model(dataset_dir, preprocess_fn, channel_list, logger, dataset_type):
    """Train a single dataset model with pooled subjects.
    
    Parameters
    ----------
    dataset_dir : str
        Path to dataset directory
    preprocess_fn : OddballPreprocessor
        Preprocessor instance
    channel_list : list
        List of channel names
    logger : logging.Logger
        Logger for experiment tracking
    dataset_type : str
        Type of dataset ('P3' or 'AVO')
        
    Returns
    -------
    dict
        Dictionary of final accuracies per subject
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_data, all_labels = [], []
    subject_ranges, subject_ids = [], []
    start_idx = 0

    # Process subjects based on dataset type
    if dataset_type == 'P3':
        for subject_dir in sorted(os.listdir(dataset_dir)):
            if not subject_dir.startswith('sub-'):
                continue
            data, labels = process_subject_data(subject_dir, dataset_dir, preprocess_fn, logger, dataset_type='P3')
            if data is None or labels is None:
                continue
            if labels.ndim > 1:
                labels = np.argmax(labels, axis=1)
            labels = labels.squeeze()
            all_data.append(data)
            all_labels.append(labels)
            end_idx = start_idx + len(data)
            subject_ranges.append((start_idx, end_idx))
            subject_ids.append(subject_dir)
            start_idx = end_idx
    elif dataset_type == 'AVO':
        avo_dataset = EEGBIDSDataset(data_dir=dataset_dir, dataset='ds005863')
        all_files = [str(f) for f in avo_dataset.get_files()]
        all_subjects = sorted(list(set([f.split('sub-')[1][:3] for f in all_files if 'sub-' in f])))
        for subject_id in all_subjects:
            data, labels = process_subject_data(subject_id, avo_dataset, preprocess_fn, logger, dataset_type='AVO')
            if data is None or labels is None:
                continue
            if labels.ndim > 1:
                labels = np.argmax(labels, axis=1)
            labels = labels.squeeze()
            all_data.append(data)
            all_labels.append(labels)
            end_idx = start_idx + len(data)
            subject_ranges.append((start_idx, end_idx))
            subject_ids.append(f"sub-{subject_id}")
            start_idx = end_idx

    if not all_data:
        logger.error(f"No data available for pooled model training in {dataset_type} dataset.")
        return {}

    # Combine data and create splits
    all_data = np.concatenate(all_data)
    all_labels = np.concatenate(all_labels)
    
    train_idx, temp_idx = train_test_split(
        range(len(all_data)), test_size=0.4, stratify=all_labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=all_labels[temp_idx]
    )
    
    train_idx, val_idx, test_idx = map(np.array, (train_idx, val_idx, test_idx))
    
    # Create data loaders
    train_loader = DataLoader(TensorDataset(
        torch.FloatTensor(all_data[train_idx]),
        torch.LongTensor(all_labels[train_idx])
    ), batch_size=BATCH_SIZE, shuffle=True)
    
    val_loader = DataLoader(TensorDataset(
        torch.FloatTensor(all_data[val_idx]),
        torch.LongTensor(all_labels[val_idx])
    ), batch_size=BATCH_SIZE, shuffle=False)
    
    test_loader = DataLoader(TensorDataset(
        torch.FloatTensor(all_data[test_idx]),
        torch.LongTensor(all_labels[test_idx])
    ), batch_size=BATCH_SIZE, shuffle=False)
    
    # Train models with different seeds
    model_accuracies = {}
    for seed in seeds:
        print(f"Training pooled model with seed {seed} ...", flush=True)
        acc_seed, model = run_experiment_with_seed(
            train_loader, val_loader, test_loader, len(channel_list), device, seed, 
            classifier, print_model_summary=(seed == seeds[0])
        )
        
        # Evaluate per subject
        subject_acc = {}
        is_lda = classifier.lower() == 'lda'
        for idx, (s_start, s_end) in enumerate(subject_ranges):
            mask = (test_idx >= s_start) & (test_idx < s_end)
            subj_indices = test_idx[mask]
            if len(subj_indices) == 0:
                continue
            
            if is_lda:
                X_subj = all_data[subj_indices].reshape(len(subj_indices), -1)
                y_subj = all_labels[subj_indices]
                lda_predictions = model.predict(X_subj)
                acc = np.mean(lda_predictions == y_subj)
                subject_acc[subject_ids[idx]] = acc
            else:
                X_subj = torch.FloatTensor(all_data[subj_indices])
                y_subj = torch.LongTensor(all_labels[subj_indices])
                subj_loader = DataLoader(TensorDataset(X_subj, y_subj), batch_size=BATCH_SIZE, shuffle=False)
                with torch.no_grad():
                    acc = evaluate(model, subj_loader, device)
                    subject_acc[subject_ids[idx]] = acc
        
        model_accuracies[f'seed_{seed}'] = subject_acc
    
    # Average across seeds
    final_acc = {}
    for sid in subject_ids:
        accs = [model_accuracies[f'seed_{s}'].get(sid, 0) for s in seeds]
        if accs:
            final_acc[sid] = np.mean(accs)
    
    return final_acc


def run_separate_subject_experiments(dataset_dir, channels, logger, dataset_type):
    """Run separate experiments for each subject.
    
    Parameters
    ----------
    dataset_dir : str
        Path to dataset directory
    channels : list
        List of channel names
    logger : logging.Logger
        Logger for experiment tracking
    dataset_type : str
        Type of dataset ('P3' or 'AVO')
        
    Returns
    -------
    dict
        Dictionary of accuracies per subject
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocessor = OddballPreprocessor(channels)
    accuracies = {}
    
    if dataset_type == 'P3':
        subject_list = sorted([d for d in os.listdir(dataset_dir) if d.startswith('sub-')])
    else:  # AVO
        avo_dataset = EEGBIDSDataset(data_dir=dataset_dir, dataset='ds005863')
        all_files = [str(f) for f in avo_dataset.get_files()]
        subject_list = sorted(list(set([f.split('sub-')[1][:3] for f in all_files if 'sub-' in f])))
    
    for i, subject in enumerate(subject_list):
        if dataset_type == 'P3':
            subject_dir = subject
            data, labels = process_subject_data(subject_dir, dataset_dir, preprocessor, logger, dataset_type='P3')
        else:  # AVO
            subject_id = subject
            avo_dataset = EEGBIDSDataset(data_dir=dataset_dir, dataset='ds005863')
            data, labels = process_subject_data(subject_id, avo_dataset, preprocessor, logger, dataset_type='AVO')
        
        if data is None:
            continue
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(data, labels)
        
        # Train with multiple seeds
        subject_accuracies_seed = [
            run_experiment_with_seed(
                train_loader, val_loader, test_loader, len(channels), device, seed, 
                classifier, print_model_summary=(i == 0)
            )[0] for seed in seeds
        ]
        
        key = subject_dir if dataset_type == 'P3' else f"sub-{subject_id}"
        accuracies[key] = np.mean(subject_accuracies_seed)
        log_individual_results(logger, dataset_type, key, accuracies[key])
    
    return accuracies 