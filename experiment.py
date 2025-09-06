"""
Experiment logic for EEG experiments
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from eegdash.data_utils import EEGBIDSDataset
from datetime import datetime
from config import LOG_DIR


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
from experiment_logger import (
    log_error, log_individual_results, log_section_header, 
    log_detailed_results, log_overall_metrics
)
# Confusion matrix plotting removed


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
        results = _run_separate_training(datasets, channels, logger, device, 
                                       p3_dir, avo_dir, exp_classifier, exp_seeds)
    else:
        # Pooled training mode: all selected datasets' subjects train one combined model
        results = _run_pooled_training(datasets, channels, logger, device,
                                     p3_dir, avo_dir, exp_classifier, exp_seeds)
    
    # Unpack results - handle both separate and pooled modes
    if len(results) == 6:  # Pooled mode returns probabilities
        accuracies, trial_counts, prediction_details, true_labels, predictions, probabilities = results
        overall_probabilities = np.array(probabilities) if probabilities else None
    else:  # Separate mode
        accuracies, trial_counts, prediction_details, true_labels, predictions = results
        overall_probabilities = None
    
    # Confusion matrix plotting removed as requested
    
    return results


def _run_separate_training(datasets, channels, logger, device, p3_dir, avo_dir, exp_classifier, exp_seeds):
    """Individual training mode: each subject trains independently"""
    all_accuracies = {}
    trial_counts = {}
    prediction_details = {}
    # Collect data for confusion matrix
    all_true_labels = []
    all_predictions = []
    all_probabilities = []
    
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
            # Check if this is sub-001 to get indices for event printing
            if subject_key == 'sub-001' and dataset_type == 'P3':
                train_loader, val_loader, test_loader, train_indices, val_indices, test_indices = create_data_loaders(
                    data, labels, return_indices=True
                )
            else:
                train_loader, val_loader, test_loader = create_data_loaders(data, labels)
            
            # Track trial counts for this subject
            final_key = f"{dataset_type}_{subject_key}" if len(datasets) > 1 else subject_key
            trial_counts[final_key] = {
                'train': len(train_loader.dataset),
                'val': len(val_loader.dataset),
                'test': len(test_loader.dataset)
            }
            
            
            # Multi-seed training
            subject_accuracies_seed = []
            subject_details_seed = []
            subject_predictions_all = []
            subject_true_labels_all = []
            
            for seed in exp_seeds:
                details, model = run_experiment_with_seed(
                    train_loader, val_loader, test_loader, len(channels), device, seed, 
                    exp_classifier, print_model_summary=(i == 0 and seed == exp_seeds[0]),
                    return_details=True
                )
                subject_accuracies_seed.append(details['accuracy'])
                subject_details_seed.append(details)
                
                # Collect predictions and true labels for confusion matrix
                is_lda = exp_classifier.lower() == 'lda'
                if is_lda:
                    # Get predictions for test set
                    X_test = []
                    y_test = []
                    for batch_data in test_loader:
                        if len(batch_data) == 3:
                            batch_X, batch_y, _ = batch_data
                        else:
                            batch_X, batch_y = batch_data
                        X_test.append(batch_X.reshape(batch_X.shape[0], -1).numpy())
                        y_test.append(batch_y.numpy())
                    X_test = np.concatenate(X_test)
                    y_test = np.concatenate(y_test)
                    predictions = model.predict(X_test)
                    subject_predictions_all.extend(predictions)
                    subject_true_labels_all.extend(y_test)
                else:
                    # Neural network - collect predictions during evaluation
                    import torch
                    model.eval()
                    with torch.no_grad():
                        for batch_data in test_loader:
                            if len(batch_data) == 3:
                                x, y, subject_indices = batch_data
                                subject_indices = subject_indices.to(device)
                            else:
                                x, y = batch_data
                                subject_indices = None
                            
                            from models import normalize_data
                            x = normalize_data(x).to(device)
                            y = y.to(device)
                            
                            if y.ndim > 1:
                                y = torch.argmax(y, dim=1)
                            
                            if hasattr(model, 'subject_layer') and subject_indices is not None:
                                scores = model(x, subject_indices)
                            else:
                                scores = model(x)
                            
                            if scores.ndim > 2:
                                scores = scores.view(scores.size(0), -1)
                            
                            _, predicted = scores.max(1)
                            subject_predictions_all.extend(predicted.cpu().numpy())
                            subject_true_labels_all.extend(y.cpu().numpy())
            
            # Store average accuracy and aggregate prediction details
            final_key = f"{dataset_type}_{subject_key}" if len(datasets) > 1 else subject_key
            all_accuracies[final_key] = np.mean(subject_accuracies_seed)
            
            # Average the prediction details across seeds
            avg_correct = np.mean([d['correct_count'] for d in subject_details_seed])
            avg_incorrect = np.mean([d['incorrect_count'] for d in subject_details_seed])
            avg_total = np.mean([d['total_count'] for d in subject_details_seed])
            avg_precision = np.mean([d['precision'] for d in subject_details_seed])
            avg_recall = np.mean([d['recall'] for d in subject_details_seed])
            avg_f1 = np.mean([d['f1_score'] for d in subject_details_seed])
            auc_values = [d.get('auc', 0.5) for d in subject_details_seed]
            # Filter out nan values and calculate mean
            valid_auc_values = [auc for auc in auc_values if not np.isnan(auc)]
            avg_auc = np.mean(valid_auc_values) if valid_auc_values else 0.5
            
            # Print detailed metrics for each subject immediately
            print(f"Subject {final_key} Results:")
            print(f"  Accuracy: {all_accuracies[final_key]:.3%}")
            # print(f"  Precision: {avg_precision:.3f}")
            # print(f"  Recall: {avg_recall:.3f}")
            # print(f"  F1-Score: {avg_f1:.3f}")
            print(f"  AUC: {avg_auc:.3f}")
            print(f"  Correct/Total: {int(avg_correct)}/{int(avg_total)}")
            print("-" * 50)
            
            # Calculate confusion matrix metrics for first seed's predictions
            if subject_true_labels_all and subject_predictions_all:
                n_test_samples = len(subject_true_labels_all) // len(exp_seeds)
                tn, fp, fn, tp = confusion_matrix(
                    subject_true_labels_all[:n_test_samples],
                    subject_predictions_all[:n_test_samples]
                ).ravel()
            else:
                tp, tn, fp, fn = 0, 0, 0, 0

            prediction_details[final_key] = {
                'correct_count': int(round(avg_correct)),
                'incorrect_count': int(round(avg_incorrect)),
                'total_count': int(round(avg_total)),
                'precision': avg_precision,
                'recall': avg_recall,
                'f1_score': avg_f1,
                'auc': avg_auc,
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            }
            
            # Add to global lists for confusion matrix (use only first seed's predictions)
            if subject_predictions_all and subject_true_labels_all:
                # Take only first seed's worth of predictions
                n_test_samples = len(subject_true_labels_all) // len(exp_seeds)
                all_predictions.extend(subject_predictions_all[:n_test_samples])
                all_true_labels.extend(subject_true_labels_all[:n_test_samples])
            
            # log_individual_results(logger, dataset_type, final_key, all_accuracies[final_key])
    
    return all_accuracies, trial_counts, prediction_details, all_true_labels, all_predictions


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
    # Collect data for confusion matrix
    confusion_true_labels = []
    confusion_predictions = []
    confusion_probabilities = []
    
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
    
    # Create weighted sampler for training set to handle class imbalance
    train_sampler = None
    train_labels = all_labels[train_indices]
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    
    # Assign weights to each sample
    sample_weights = np.array([class_weights[label] for label in train_labels])
    sample_weights = torch.from_numpy(sample_weights).double()
    
    from torch.utils.data import WeightedRandomSampler
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    print(f"Pooled training - Class distribution: {class_counts.tolist()}, "
          f"Class weights: {class_weights.tolist()}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Multi-seed training
    model_accuracies = {}
    model_details = {}
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
        subject_details = {}
        for subject_idx, (s_start, s_end) in enumerate(subject_ranges):
            mask = (test_indices >= s_start) & (test_indices < s_end)
            subject_test_indices = test_indices[mask]
            if len(subject_test_indices) == 0:
                continue
            
            if is_lda:
                X_subj = all_data[subject_test_indices].reshape(len(subject_test_indices), -1)
                y_subj = all_labels[subject_test_indices]
                predictions = model.predict(X_subj)
                correct_count = np.sum(predictions == y_subj)
                total_count = len(y_subj)
                acc = correct_count / total_count
                
                # Calculate detailed metrics for LDA
                try:
                    # Get probability estimates for AUC calculation
                    y_proba = model.predict_proba(X_subj)[:, 1]  # Probability of positive class
                except:
                    y_proba = predictions  # Fallback to binary predictions if probabilities not available
                
                # Calculate confusion matrix metrics first
                tn, fp, fn, tp = confusion_matrix(y_subj, predictions).ravel()
                
                # Calculate precision, recall, f1 from confusion matrix
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                try:
                    # Check if we have both classes in the true labels
                    unique_labels = np.unique(y_subj)
                    if len(unique_labels) < 2:
                        print(f"Warning: Subject {subject_id} has only one class in test set: {unique_labels}. Setting AUC to 0.5.")
                        auc = 0.5
                    else:
                        # Check for problematic probability values
                        if np.any(np.isnan(y_proba)) or np.any(np.isinf(y_proba)):
                            print(f"Warning: Subject {subject_id} has NaN or infinite values in probabilities. Setting AUC to 0.5.")
                            auc = 0.5
                        else:
                            auc = roc_auc_score(y_subj, y_proba)
                            if np.isnan(auc):
                                print(f"Warning: Subject {subject_id} AUC calculation returned NaN. Setting to 0.5.")
                                auc = 0.5
                except Exception as e:
                    print(f"Warning: Subject {subject_id} AUC calculation failed: {e}. Setting to 0.5.")
                    auc = 0.5
                
                details = {
                    'accuracy': acc,
                    'correct_count': correct_count,
                    'incorrect_count': total_count - correct_count,
                    'total_count': total_count,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc': auc,
                    'tp': int(tp),
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn)
                }
                # Collect data for confusion matrix (only for first seed)
                if seed == exp_seeds[0]:
                    confusion_predictions.extend(predictions)
                    confusion_true_labels.extend(y_subj)
                    confusion_probabilities.extend(y_proba)
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
                    details = evaluate(model, subj_loader, device, return_details=True)
                    acc = details['accuracy']
                
                # Calculate confusion matrix metrics for deep learning model
                y_true = []
                y_pred = []
                model.eval()
                with torch.no_grad():
                    for batch_data in subj_loader:
                        if len(batch_data) == 3:
                            x, y, subject_indices_batch = batch_data
                            subject_indices_batch = subject_indices_batch.to(device)
                        else:
                            x, y = batch_data
                            subject_indices_batch = None
                        
                        x = normalize_data(x).to(device)
                        y = y.to(device)
                        
                        if y.ndim > 1:
                            y = torch.argmax(y, dim=1)
                        
                        if hasattr(model, 'subject_layer') and subject_indices_batch is not None:
                            scores = model(x, subject_indices_batch)
                        else:
                            scores = model(x)
                        
                        if scores.ndim > 2:
                            scores = scores.view(scores.size(0), -1)
                        
                        _, predicted = scores.max(1)
                        y_true.extend(y.cpu().numpy())
                        y_pred.extend(predicted.cpu().numpy())
                
                # Calculate confusion matrix metrics first
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                
                # Calculate precision, recall, f1 from confusion matrix
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Update metrics in details
                details.update({
                    'accuracy': acc,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'tp': int(tp),
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn)
                })
                
                # Collect predictions for confusion matrix (only for first seed)
                if seed == exp_seeds[0]:
                    model.eval()
                    with torch.no_grad():
                        for batch_data in subj_loader:
                            if len(batch_data) == 3:
                                x, y, subject_indices_batch = batch_data
                                subject_indices_batch = subject_indices_batch.to(device)
                            else:
                                x, y = batch_data
                                subject_indices_batch = None
                            
                            x = normalize_data(x).to(device)
                            y = y.to(device)
                            
                            if y.ndim > 1:
                                y = torch.argmax(y, dim=1)
                            
                            if hasattr(model, 'subject_layer') and subject_indices_batch is not None:
                                scores = model(x, subject_indices_batch)
                            else:
                                scores = model(x)
                            
                            if scores.ndim > 2:
                                scores = scores.view(scores.size(0), -1)
                            
                            _, predicted = scores.max(1)
                            confusion_predictions.extend(predicted.cpu().numpy())
                            confusion_true_labels.extend(y.cpu().numpy())
            
            subject_accuracies[subject_ids[subject_idx]] = acc
            subject_details[subject_ids[subject_idx]] = details
        
        model_accuracies[f"seed_{seed}"] = subject_accuracies
        model_details[f"seed_{seed}"] = subject_details
    
    # Cross-seed averaging
    final_accuracies = {}
    trial_counts = {}
    prediction_details = {}
    
    for subject_id in subject_ids:
        # Calculate average accuracy across seeds
        accs = [model_accuracies[f"seed_{seed}"].get(subject_id, 0) for seed in exp_seeds]
        if accs:
            final_accuracies[subject_id] = np.mean(accs)
            
        # Average prediction details across seeds
        details_list = [model_details[f"seed_{seed}"].get(subject_id, {}) for seed in exp_seeds]
        if details_list and all(d for d in details_list):
            avg_correct = np.mean([d['correct_count'] for d in details_list])
            avg_incorrect = np.mean([d['incorrect_count'] for d in details_list])
            avg_total = np.mean([d['total_count'] for d in details_list])
            avg_precision = np.mean([d.get('precision', 0) for d in details_list])
            avg_recall = np.mean([d.get('recall', 0) for d in details_list])
            avg_f1 = np.mean([d.get('f1_score', 0) for d in details_list])
            auc_values = [d.get('auc', 0.5) for d in details_list]
            # Filter out nan values and calculate mean
            valid_auc_values = [auc for auc in auc_values if not np.isnan(auc)]
            avg_auc = np.mean(valid_auc_values) if valid_auc_values else 0.5
            
            # Average confusion matrix metrics
            avg_tp = np.mean([d.get('tp', 0) for d in details_list])
            avg_tn = np.mean([d.get('tn', 0) for d in details_list])
            avg_fp = np.mean([d.get('fp', 0) for d in details_list])
            avg_fn = np.mean([d.get('fn', 0) for d in details_list])

            prediction_details[subject_id] = {
                'correct_count': int(round(avg_correct)),
                'incorrect_count': int(round(avg_incorrect)),
                'total_count': int(round(avg_total)),
                'precision': avg_precision,
                'recall': avg_recall,
                'f1_score': avg_f1,
                'auc': avg_auc,
                'tp': int(round(avg_tp)),
                'tn': int(round(avg_tn)),
                'fp': int(round(avg_fp)),
                'fn': int(round(avg_fn))
            }
            
        # Calculate trial counts for each subject
        mask_train = np.isin(train_indices, range(*subject_ranges[subject_ids.index(subject_id)]))
        mask_val = np.isin(val_indices, range(*subject_ranges[subject_ids.index(subject_id)]))
        mask_test = np.isin(test_indices, range(*subject_ranges[subject_ids.index(subject_id)]))
        
        trial_counts[subject_id] = {
            'train': np.sum(mask_train),
            'val': np.sum(mask_val),
            'test': np.sum(mask_test)
        }
    
    return final_accuracies, trial_counts, prediction_details, confusion_true_labels, confusion_predictions, confusion_probabilities


# Backward compatibility wrapper functions
def train_combined_model(p3_dir, avo_dataset, channels, logger):
    accuracies, trial_counts, prediction_details, _, _ = run_experiment(
        datasets=['P3', 'AVO'], 
        training_mode='pooled',
        channels=channels,
        logger=logger,
        p3_dir=p3_dir,
        avo_dir=avo_dataset
    )
    return accuracies, trial_counts, prediction_details


def train_single_dataset_model(dataset_dir, preprocess_fn, channel_list, logger, dataset_type):
    accuracies, _ = run_experiment(
        datasets=[dataset_type],
        training_mode='pooled', 
        channels=channel_list,
        logger=logger,
        p3_dir=dataset_dir if dataset_type == 'P3' else P3_DATA_DIR,
        avo_dir=dataset_dir if dataset_type == 'AVO' else AVO_DATA_DIR
    )
    return accuracies


def run_separate_subject_experiments(dataset_dir, channels, logger, dataset_type):
    accuracies, _ = run_experiment(
        datasets=[dataset_type],
        training_mode='separate',
        channels=channels, 
        logger=logger,
        p3_dir=dataset_dir if dataset_type == 'P3' else P3_DATA_DIR,
        avo_dir=dataset_dir if dataset_type == 'AVO' else AVO_DATA_DIR
    )
    return accuracies
