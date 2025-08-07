"""
Experiment logic for EEG experiments
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from eegdash.data_utils import EEGBIDSDataset


class SubjectDataset(Dataset):
    """Dataset that includes subject indices for each sample."""
    def __init__(self, data, labels, subject_indices):
        self.data = data
        self.labels = labels
        self.subject_indices = subject_indices
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.subject_indices[idx]

from config import (
    P3_DATA_DIR, AVO_DATA_DIR, BATCH_SIZE, seeds, 
    use_combined_datasets, separate_subject_classification, 
    electrode_list, classifier, VAL_SIZE, TEST_SIZE, use_subject_layer
)
from constants import COMMON_CHANNELS, P3_CHANNELS, AVO_CHANNELS
from preprocessor import OddballPreprocessor
from models import create_model, train_model, evaluate, normalize_data
from utils import run_experiment_with_seed, create_data_loaders, calculate_statistics, print_statistics, process_subject_data
from experiment_logger import log_error, log_individual_results, log_section_header


def get_dataset_subjects(dataset_type, dataset_obj):
    if dataset_type == 'P3':
        return sorted([d for d in os.listdir(dataset_obj) if d.startswith('sub-')])
    elif dataset_type == 'AVO':
        all_files = [str(f) for f in dataset_obj.get_files()]
        return sorted(list(set([f.split('sub-')[1][:3] for f in all_files if 'sub-' in f])))
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


def process_dataset_subjects(dataset_info, dataset_type, prefix, channels, logger,
                           all_data, all_labels, subject_ranges, subject_ids, start_idx):
    """
    Process subjects from a single dataset.
    """
    dataset_obj, subject_list = dataset_info
    preprocessor = OddballPreprocessor(channels)
    
    for subject_id in subject_list:
        print(f"Loading {dataset_type} subject {subject_id} ...", flush=True)
        data, labels = process_subject_data(subject_id, dataset_obj, preprocessor, logger, dataset_type=dataset_type)
        
        if data is not None and labels is not None:
            # Standardize label format
            if labels.ndim > 1:
                labels = np.argmax(labels, axis=1)
            labels = labels.squeeze()
            
            # Add to combined dataset
            all_data.append(data)
            all_labels.append(labels)
            end_idx = start_idx + len(data)
            subject_ranges.append((start_idx, end_idx))
            subject_ids.append(f"{prefix}_{subject_id}")
            start_idx = end_idx
    
    return start_idx


def process_dataset_subjects_with_indices(dataset_info, dataset_type, prefix, channels, logger,
                           all_data, all_labels, all_subject_indices, subject_ranges, subject_ids, 
                           subject_id_to_index, start_idx, current_subject_index):
    """
    Process subjects from a single dataset with subject indices for subject layer.
    """
    dataset_obj, subject_list = dataset_info
    preprocessor = OddballPreprocessor(channels)
    
    for subject_id in subject_list:
        print(f"Loading {dataset_type} subject {subject_id} ...", flush=True)
        data, labels = process_subject_data(subject_id, dataset_obj, preprocessor, logger, dataset_type=dataset_type)
        
        if data is not None and labels is not None:
            # Standardize label format
            if labels.ndim > 1:
                labels = np.argmax(labels, axis=1)
            labels = labels.squeeze()
            
            # Create subject identifier
            full_subject_id = f"{prefix}_{subject_id}" if prefix else subject_id
            
            # Assign subject index
            if full_subject_id not in subject_id_to_index:
                subject_id_to_index[full_subject_id] = current_subject_index
                current_subject_index += 1
            
            subject_index = subject_id_to_index[full_subject_id]
            
            # Create subject indices array for all samples from this subject
            subject_indices = np.full(len(data), subject_index, dtype=np.int64)
            
            # Add to combined dataset
            all_data.append(data)
            all_labels.append(labels)
            all_subject_indices.append(subject_indices)
            end_idx = start_idx + len(data)
            subject_ranges.append((start_idx, end_idx))
            subject_ids.append(full_subject_id)
            start_idx = end_idx
    
    return start_idx, current_subject_index


def run_experiment(datasets, training_mode, channels, logger, **kwargs):
    """
    Unified experiment training function with parameter-controlled experiment configurations
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get configuration from kwargs
    p3_dir = kwargs.get('p3_dir', P3_DATA_DIR)
    avo_dir = kwargs.get('avo_dir', AVO_DATA_DIR) 
    exp_classifier = kwargs.get('classifier', classifier)
    exp_seeds = kwargs.get('seeds', seeds)
    
    if training_mode == 'separate':
        # Individual training mode: each subject trains a separate model
        return _run_separate_training(datasets, channels, logger, device, 
                                    p3_dir, avo_dir, exp_classifier, exp_seeds)
    else:
        # Pooled training mode: all selected datasets' subjects train one combined model
        return _run_pooled_training(datasets, channels, logger, device,
                                  p3_dir, avo_dir, exp_classifier, exp_seeds)


def _run_separate_training(datasets, channels, logger, device, p3_dir, avo_dir, exp_classifier, exp_seeds):
    """Individual training mode: each subject trains independently"""
    all_accuracies = {}
    
    for dataset_type in datasets:
        if dataset_type == 'P3':
            dataset_dir = p3_dir
            subject_list = get_dataset_subjects('P3', p3_dir)
        elif dataset_type == 'AVO':
            dataset_dir = avo_dir
            avo_dataset = EEGBIDSDataset(data_dir=avo_dir, dataset='ds005863')
            subject_list = get_dataset_subjects('AVO', avo_dataset)
        
        preprocessor = OddballPreprocessor(channels)
        
        for i, subject in enumerate(subject_list):
            if dataset_type == 'P3':
                data, labels = process_subject_data(subject, dataset_dir, preprocessor, logger, dataset_type='P3')
                subject_key = subject
            else:  # AVO
                data, labels = process_subject_data(subject, avo_dataset, preprocessor, logger, dataset_type='AVO')
                subject_key = f"sub-{subject}"
            
            if data is None:
                continue
            
            # Create data loaders for the current subject
            train_loader, val_loader, test_loader = create_data_loaders(data, labels)
            
            # Multi-seed training
            subject_accuracies_seed = []
            for seed in exp_seeds:
                acc, _ = run_experiment_with_seed(
                    train_loader, val_loader, test_loader, len(channels), device, seed, 
                    exp_classifier, print_model_summary=(i == 0 and seed == exp_seeds[0])
                )
                subject_accuracies_seed.append(acc)
            
            # Store average accuracy
            final_key = f"{dataset_type}_{subject_key}" if len(datasets) > 1 else subject_key
            all_accuracies[final_key] = np.mean(subject_accuracies_seed)
            log_individual_results(logger, dataset_type, final_key, all_accuracies[final_key])
    
    return all_accuracies


def _run_pooled_training(datasets, channels, logger, device, p3_dir, avo_dir, exp_classifier, exp_seeds):
    """Pooled training mode: all subject data combined to train one model"""
    all_data = []
    all_labels = []
    all_subject_indices = []
    subject_ranges = []
    subject_ids = []
    subject_id_to_index = {}  # Map subject_id to numeric index
    start_idx = 0
    current_subject_index = 0
    
    # Collect data from all specified datasets
    for dataset_type in datasets:
        if dataset_type == 'P3':
            subjects = get_dataset_subjects('P3', p3_dir)
            prefix = 'P3' if len(datasets) > 1 else ''
            start_idx, current_subject_index = process_dataset_subjects_with_indices(
                (p3_dir, subjects), dataset_type, prefix, 
                channels, logger, all_data, all_labels, all_subject_indices, 
                subject_ranges, subject_ids, subject_id_to_index, start_idx, current_subject_index
            )
        elif dataset_type == 'AVO':
            avo_dataset = EEGBIDSDataset(data_dir=avo_dir, dataset='ds005863')
            subjects = get_dataset_subjects('AVO', avo_dataset)
            prefix = 'AVO' if len(datasets) > 1 else 'sub'
            start_idx, current_subject_index = process_dataset_subjects_with_indices(
                (avo_dataset, subjects), dataset_type, prefix,
                channels, logger, all_data, all_labels, all_subject_indices,
                subject_ranges, subject_ids, subject_id_to_index, start_idx, current_subject_index
            )
    
    if not all_data:
        logger.error("No data available for training")
        return {}
    
    # Combine all data
    all_data = np.concatenate(all_data)
    all_labels = np.concatenate(all_labels)
    all_subject_indices = np.concatenate(all_subject_indices)
    
    # Create data splits
    temp_size = VAL_SIZE + TEST_SIZE
    train_indices, temp_indices = train_test_split(
        range(len(all_data)), test_size=temp_size, stratify=all_labels
    )
    test_ratio = TEST_SIZE / temp_size
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=test_ratio, stratify=all_labels[temp_indices]
    )
    
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    # Determine whether to use subject layer
    n_subjects = len(subject_id_to_index)
    should_use_subject_layer = (use_subject_layer and 
                               exp_classifier == 'ShallowFBCSPNet' and 
                               not separate_subject_classification and
                               n_subjects > 1)
    
    # Create data loaders
    if should_use_subject_layer:
        train_dataset = SubjectDataset(
            torch.FloatTensor(all_data[train_indices]), 
            torch.LongTensor(all_labels[train_indices]),
            torch.LongTensor(all_subject_indices[train_indices])
        )
        val_dataset = SubjectDataset(
            torch.FloatTensor(all_data[val_indices]), 
            torch.LongTensor(all_labels[val_indices]),
            torch.LongTensor(all_subject_indices[val_indices])
        )
        test_dataset = SubjectDataset(
            torch.FloatTensor(all_data[test_indices]), 
            torch.LongTensor(all_labels[test_indices]),
            torch.LongTensor(all_subject_indices[test_indices])
        )
    else:
        train_dataset = TensorDataset(torch.FloatTensor(all_data[train_indices]), torch.LongTensor(all_labels[train_indices]))
        val_dataset = TensorDataset(torch.FloatTensor(all_data[val_indices]), torch.LongTensor(all_labels[val_indices]))
        test_dataset = TensorDataset(torch.FloatTensor(all_data[test_indices]), torch.LongTensor(all_labels[test_indices]))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Multi-seed training
    model_accuracies = {}
    for seed in exp_seeds:
        print(f"Training pooled model (datasets: {datasets}) with seed {seed} ...", flush=True)
        
        is_lda = exp_classifier.lower() == 'lda'
        if is_lda:
            # LDA training
            np.random.seed(seed)
            X_train = []
            y_train = []
            for batch_data in train_loader:
                if len(batch_data) == 3:  # (X, y, subject_indices)
                    batch_X, batch_y, _ = batch_data
                else:  # (X, y)
                    batch_X, batch_y = batch_data
                X_train.append(batch_X.reshape(batch_X.shape[0], -1).numpy())
                y_train.append(batch_y.numpy())
            X_train = np.concatenate(X_train)
            y_train = np.concatenate(y_train)
            
            model = create_model(len(channels), is_lda=True)
            model.fit(X_train, y_train)
        else:
            # Neural network training
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Create model with subject layer if enabled
            model = create_model(
                len(channels), 
                is_lda=False, 
                n_subjects=n_subjects if should_use_subject_layer else None,
                enable_subject_layer=should_use_subject_layer
            )
            model = model.to(device)
            
            if seed == exp_seeds[0]:
                print(f"\nModel Architecture Summary (Datasets: {datasets})")
                print("="*60)
                print(f"Model type: {type(model).__name__}")
                print(f"Input channels: {len(channels)}")
                print(f"Number of subjects: {n_subjects}")
                print(f"Subject layer enabled: {should_use_subject_layer}")
                print(f"Input shape: (batch_size, {len(channels)}, 128)")
                print("="*60 + "\n")
            
            train_model(model, train_loader, val_loader, test_loader, device, is_lda=False)
        
        # Evaluate each subject
        subject_accuracies = {}
        for subject_idx, (s_start, s_end) in enumerate(subject_ranges):
            mask = (test_indices >= s_start) & (test_indices < s_end)
            subject_test_indices = test_indices[mask]
            if len(subject_test_indices) == 0:
                continue
            
            if is_lda:
                X_subj = all_data[subject_test_indices].reshape(len(subject_test_indices), -1)
                y_subj = all_labels[subject_test_indices]
                predictions = model.predict(X_subj)
                acc = np.mean(predictions == y_subj)
            else:
                X_subj = torch.FloatTensor(all_data[subject_test_indices])
                y_subj = torch.LongTensor(all_labels[subject_test_indices])
                
                if should_use_subject_layer:
                    # Include subject indices for evaluation
                    subj_indices = torch.LongTensor(all_subject_indices[subject_test_indices])
                    subj_dataset = SubjectDataset(X_subj, y_subj, subj_indices)
                else:
                    subj_dataset = TensorDataset(X_subj, y_subj)
                
                subj_loader = DataLoader(subj_dataset, batch_size=BATCH_SIZE, shuffle=False)
                with torch.no_grad():
                    acc = evaluate(model, subj_loader, device)
            
            subject_accuracies[subject_ids[subject_idx]] = acc
        
        model_accuracies[f"seed_{seed}"] = subject_accuracies
    
    # Cross-seed averaging
    final_accuracies = {}
    for subject_id in subject_ids:
        accs = [model_accuracies[f"seed_{seed}"].get(subject_id, 0) for seed in exp_seeds]
        if accs:
            final_accuracies[subject_id] = np.mean(accs)
    
    return final_accuracies


# Backward compatibility wrapper functions
def train_combined_model(p3_dir, avo_dataset, channels, logger):
    return run_experiment(
        datasets=['P3', 'AVO'], 
        training_mode='pooled',
        channels=channels,
        logger=logger,
        p3_dir=p3_dir,
        avo_dir=avo_dataset
    )


def train_single_dataset_model(dataset_dir, preprocess_fn, channel_list, logger, dataset_type):
    return run_experiment(
        datasets=[dataset_type],
        training_mode='pooled', 
        channels=channel_list,
        logger=logger,
        p3_dir=dataset_dir if dataset_type == 'P3' else P3_DATA_DIR,
        avo_dir=dataset_dir if dataset_type == 'AVO' else AVO_DATA_DIR
    )


def run_separate_subject_experiments(dataset_dir, channels, logger, dataset_type):
    return run_experiment(
        datasets=[dataset_type],
        training_mode='separate',
        channels=channels, 
        logger=logger,
        p3_dir=dataset_dir if dataset_type == 'P3' else P3_DATA_DIR,
        avo_dir=dataset_dir if dataset_type == 'AVO' else AVO_DATA_DIR
    ) 