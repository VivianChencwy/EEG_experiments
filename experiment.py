"""
Experiment logic for EEG experiments
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from data_utils import EEGBIDSDataset
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
    electrode_list, classifier, VAL_SIZE, TEST_SIZE, use_subject_layer,
    LEARNING_RATE, WEIGHT_DECAY, DROPOUT_RATE, MAX_EPOCHS, EARLY_STOPPING_PATIENCE,
    USE_DATA_AUGMENTATION, NOISE_STD, TIME_SHIFT_RANGE, LABEL_SMOOTHING,
    USE_ENHANCED_PREPROCESSING, REMOVE_ARTIFACTS, BASELINE_CORRECT,
    EXTRACT_FREQUENCY_FEATURES, APPLY_NOTCH_FILTER,
    ELECTRODE_FUSION_METHOD, DOMAIN_ADAPTATION_METHOD,
    ENABLE_COMPREHENSIVE_EVALUATION, ENABLE_DOMAIN_ANALYSIS,
    DEVICE_MODE, USE_NESTED_CV, NESTED_CV_OUTER_FOLDS, NESTED_CV_INNER_FOLDS,
    NESTED_CV_REPEATS, NESTED_CV_CONFIDENCE_LEVEL
)
from constants import COMMON_CHANNELS, P3_CHANNELS, AVO_CHANNELS
from preprocessor import OddballPreprocessor
from enhanced_preprocessor import EnhancedOddballPreprocessor
from models import create_model, train_model, evaluate, normalize_data, create_fusion_model, train_fusion_model
from enhanced_preprocessor import FusionDatasetManager
from evaluation_utils import ComprehensiveEvaluator, CrossDatasetEvaluator, ResultsComparator
from utils import run_experiment_with_seed, create_data_loaders, calculate_statistics, print_statistics, process_subject_data
from experiment_logger import (
    setup_logger, log_error, log_individual_results, log_section_header,
    log_detailed_results, log_overall_metrics
)
from nested_cv import NestedCrossValidation, run_nested_cv_experiment
# Confusion matrix plotting removed


def get_device():
    """根据配置获取设备"""
    if DEVICE_MODE == 'cpu':
        return torch.device('cpu')
    elif DEVICE_MODE == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device('cuda')
    else:  # 'auto'
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_preprocessor(channels, dataset_type):
    """Create the appropriate preprocessor based on configuration."""
    if USE_ENHANCED_PREPROCESSING:
        print("Using Enhanced Preprocessor with advanced features:")
        print(f"  - Artifact removal (ICA): {REMOVE_ARTIFACTS}")
        print(f"  - Baseline correction: {BASELINE_CORRECT}")
        print(f"  - Frequency features: {EXTRACT_FREQUENCY_FEATURES}")
        print(f"  - Notch filter: {APPLY_NOTCH_FILTER}")

        return EnhancedOddballPreprocessor(
            channels,
            dataset_type=dataset_type,
            remove_artifacts=REMOVE_ARTIFACTS,
            baseline_correct=BASELINE_CORRECT,
            extract_frequency_features=EXTRACT_FREQUENCY_FEATURES,
            apply_notch_filter=APPLY_NOTCH_FILTER
        )
    else:
        print("Using Standard Preprocessor")
        return OddballPreprocessor(channels, dataset_type=dataset_type)


def get_dataset_subjects(dataset_type, dataset_obj):
    from config import MAX_SUBJECTS_P3, MAX_SUBJECTS_AVO
    
    if dataset_type == 'P3':
        all_subjects = sorted([d for d in os.listdir(dataset_obj) if d.startswith('sub-')])
        # Limit P3 dataset to configured maximum
        if MAX_SUBJECTS_P3 is not None:
            return all_subjects[:MAX_SUBJECTS_P3]
        return all_subjects
    elif dataset_type == 'AVO':
        all_files = [str(f) for f in dataset_obj.get_files()]
        all_subjects = sorted(list(set([f.split('sub-')[1][:3] for f in all_files if 'sub-' in f])))
        # Limit AVO dataset to configured maximum
        if MAX_SUBJECTS_AVO is not None:
            return all_subjects[:MAX_SUBJECTS_AVO]
        return all_subjects
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


def process_dataset_subjects(dataset_info, dataset_type, prefix, channels, logger,
                           all_data, all_labels, subject_ranges, subject_ids, start_idx):
    """
    Process subjects from a single dataset.
    """
    dataset_obj, subject_list = dataset_info
    preprocessor = create_preprocessor(channels, dataset_type)
    
    for subject_id in subject_list:
        # Processing subject silently
        data, labels = process_subject_data(subject_id, dataset_obj, preprocessor, logger, dataset_type=dataset_type)
        
        if data is not None and labels is not None:
            # Print original trial count for this subject
            full_subject_id = f"{prefix}_{subject_id}" if prefix else subject_id
            print(f"Subject {full_subject_id} ({dataset_type}): {len(data)} total trials available")
            
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
    preprocessor = create_preprocessor(channels, dataset_type)
    
    for subject_id in subject_list:
        # Processing subject silently
        data, labels = process_subject_data(subject_id, dataset_obj, preprocessor, logger, dataset_type=dataset_type)
        
        if data is not None and labels is not None:
            # Print original trial count for this subject
            full_subject_id = f"{prefix}_{subject_id}" if prefix else subject_id
            print(f"Subject {full_subject_id} ({dataset_type}): {len(data)} total trials available")
            
            # Standardize label format
            if labels.ndim > 1:
                labels = np.argmax(labels, axis=1)
            labels = labels.squeeze()
            
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
    device = get_device()

    # Get configuration from kwargs
    p3_dir = kwargs.get('p3_dir', P3_DATA_DIR)
    avo_dir = kwargs.get('avo_dir', AVO_DATA_DIR)
    exp_classifier = kwargs.get('classifier', classifier)
    exp_seeds = kwargs.get('seeds', seeds)

    # Check if nested CV is enabled
    if USE_NESTED_CV:
        logger.info("Using Nested Cross-Validation (politically correct approach)")
        logger.info(f"Configuration: {NESTED_CV_OUTER_FOLDS}-fold outer, {NESTED_CV_INNER_FOLDS}-fold inner, {NESTED_CV_REPEATS} repeats")
        return _run_nested_cv_experiment(datasets, channels, logger, device,
                                       p3_dir, avo_dir, exp_classifier, exp_seeds)
    else:
        logger.info("Using traditional train/validation/test split")
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


def _run_nested_cv_experiment(datasets, channels, logger, device, p3_dir, avo_dir, exp_classifier, exp_seeds):
    """
    Run nested cross-validation experiment with aggregated data from all specified datasets.
    """
    logger.info("="*60)
    logger.info("NESTED CROSS-VALIDATION CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Outer folds: {NESTED_CV_OUTER_FOLDS}")
    logger.info(f"Inner folds: {NESTED_CV_INNER_FOLDS}")
    logger.info(f"Repeats: {NESTED_CV_REPEATS}")
    logger.info(f"Confidence level: {NESTED_CV_CONFIDENCE_LEVEL}")
    logger.info("="*60)

    # Collect all data from specified datasets
    all_data = []
    all_labels = []
    all_subject_indices = []
    subject_ranges = []
    subject_ids = []
    subject_id_to_index = {}
    start_idx = 0
    current_subject_index = 0

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
            from data_utils import EEGBIDSDataset
            avo_dataset = EEGBIDSDataset(data_dir=avo_dir, dataset='ds005863')
            subjects = get_dataset_subjects('AVO', avo_dataset)
            prefix = 'AVO' if len(datasets) > 1 else 'sub'
            start_idx, current_subject_index = process_dataset_subjects_with_indices(
                (avo_dataset, subjects), dataset_type, prefix,
                channels, logger, all_data, all_labels, all_subject_indices,
                subject_ranges, subject_ids, subject_id_to_index, start_idx, current_subject_index
            )

    if not all_data:
        logger.error("No data available for nested CV")
        return {}, {}, {}, [], []

    # Combine all data
    all_data = np.concatenate(all_data)
    all_labels = np.concatenate(all_labels)
    all_subject_indices = np.concatenate(all_subject_indices)

    logger.info(f"Nested CV dataset summary:")
    logger.info(f"  Total subjects: {len(subject_ids)}")
    logger.info(f"  Total trials: {len(all_data)}")
    logger.info(f"  Average trials per subject: {len(all_data) / len(subject_ids):.1f}")

    # Detect actual input channels
    actual_input_channels = all_data.shape[1]
    if actual_input_channels != len(channels):
        logger.info(f"Enhanced preprocessing increased channels from {len(channels)} to {actual_input_channels}")

    # Run nested cross-validation
    nested_cv = NestedCrossValidation(
        outer_cv_folds=NESTED_CV_OUTER_FOLDS,
        inner_cv_folds=NESTED_CV_INNER_FOLDS,
        n_repeats=NESTED_CV_REPEATS,
        random_state=42,
        logger=logger
    )

    nested_results = nested_cv.run_nested_cv(
        data=all_data,
        labels=all_labels,
        model_name=exp_classifier,
        n_channels=len(channels),
        device=device,
        subject_indices=all_subject_indices
    )

    # Convert nested CV results to format compatible with existing code
    accuracies = {}
    prediction_details = {}
    trial_counts = {}

    # Extract per-subject performance from nested CV results
    # For nested CV, we report overall performance rather than per-subject
    overall_acc = nested_results['mean_accuracy']
    overall_ci_lower = nested_results['ci_lower']
    overall_ci_upper = nested_results['ci_upper']

    # Create summary results for each subject (using overall performance as estimate)
    for subject_id in subject_ids:
        accuracies[subject_id] = overall_acc
        prediction_details[subject_id] = {
            'accuracy': overall_acc,
            'confidence_interval': (overall_ci_lower, overall_ci_upper),
            'nested_cv_results': nested_results,
            'precision': nested_results['other_metrics'].get('precision', {}).get('mean', 0.0),
            'recall': nested_results['other_metrics'].get('recall', {}).get('mean', 0.0),
            'f1_score': nested_results['other_metrics'].get('f1_score', {}).get('mean', 0.0),
            'auc': nested_results['other_metrics'].get('auc', {}).get('mean', 0.5)
        }

        # Calculate trial counts for this subject
        subject_idx = subject_ids.index(subject_id)
        subject_start, subject_end = subject_ranges[subject_idx]
        subject_trials = subject_end - subject_start

        trial_counts[subject_id] = {
            'total': subject_trials,
            'nested_cv_folds': f"{NESTED_CV_OUTER_FOLDS}x{NESTED_CV_INNER_FOLDS}",
            'repeats': NESTED_CV_REPEATS
        }

    # Log nested CV results
    logger.info("\n" + "="*60)
    logger.info("NESTED CROSS-VALIDATION RESULTS")
    logger.info("="*60)
    logger.info(f"Model: {exp_classifier}")
    logger.info(f"Overall accuracy: {overall_acc:.4f} ± {nested_results['std_accuracy']:.4f}")
    logger.info(f"95% Confidence Interval: [{overall_ci_lower:.4f}, {overall_ci_upper:.4f}]")
    logger.info(f"Statistical significance: {nested_results['statistical_significance']['n_samples']} evaluations")

    # Log other metrics with confidence intervals
    for metric_name, metric_stats in nested_results['other_metrics'].items():
        logger.info(f"{metric_name.title()}: {metric_stats['mean']:.4f} "
                   f"[{metric_stats['ci_lower']:.4f}, {metric_stats['ci_upper']:.4f}]")

    logger.info("="*60)

    # Return in format compatible with existing experiment flow
    return accuracies, trial_counts, prediction_details, [], []


def _run_separate_training(datasets, channels, logger, device, p3_dir, avo_dir, exp_classifier, exp_seeds):
    """Individual training mode: each subject trains independently"""
    # Print configuration information
    from config import (
        TRAIN_SIZE, VAL_SIZE, TEST_SIZE, MAX_SUBJECTS_P3, MAX_SUBJECTS_AVO,
        MAX_TRIALS_PER_SUBJECT_TRAIN, MAX_TRIALS_PER_SUBJECT_VAL, MAX_TRIALS_PER_SUBJECT_TEST,
        FIXED_TRIALS_PER_SUBJECT_TRAIN, FIXED_TRIALS_PER_SUBJECT_VAL, FIXED_TRIALS_PER_SUBJECT_TEST
    )
    
    print("="*60)
    print("SEPARATE TRAINING CONFIGURATION")
    print("="*60)
    print(f"Datasets: {datasets}")
    print(f"Max subjects - P3: {MAX_SUBJECTS_P3}, AVO: {MAX_SUBJECTS_AVO}")
    
    # Check trial limits
    if any(x is not None for x in [FIXED_TRIALS_PER_SUBJECT_TRAIN, FIXED_TRIALS_PER_SUBJECT_VAL, FIXED_TRIALS_PER_SUBJECT_TEST]):
        print("Fixed trial counts per subject:")
        print(f"  Train: {FIXED_TRIALS_PER_SUBJECT_TRAIN}, Val: {FIXED_TRIALS_PER_SUBJECT_VAL}, Test: {FIXED_TRIALS_PER_SUBJECT_TEST}")
        print("Using fixed trial counts instead of ratio-based splitting")
    elif any(x is not None for x in [MAX_TRIALS_PER_SUBJECT_TRAIN, MAX_TRIALS_PER_SUBJECT_VAL, MAX_TRIALS_PER_SUBJECT_TEST]):
        print("Max trial limits per subject:")
        print(f"  Train: {MAX_TRIALS_PER_SUBJECT_TRAIN}, Val: {MAX_TRIALS_PER_SUBJECT_VAL}, Test: {MAX_TRIALS_PER_SUBJECT_TEST}")
        print(f"Train/Val/Test split: {TRAIN_SIZE:.1f}:{VAL_SIZE:.1f}:{TEST_SIZE:.1f}")
    else:
        print("No trial limits - using all available trials per subject")
        print(f"Train/Val/Test split: {TRAIN_SIZE:.1f}:{VAL_SIZE:.1f}:{TEST_SIZE:.1f}")
    print("="*60)
    print()
    
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
        
        
        preprocessor = create_preprocessor(channels, dataset_type)
        
        for i, subject in enumerate(subject_list):
            if dataset_type == 'P3':
                data, labels = process_subject_data(subject, dataset_dir, preprocessor, logger, dataset_type='P3')
                subject_key = subject
            else:  # AVO
                data, labels = process_subject_data(subject, avo_dataset, preprocessor, logger, dataset_type='AVO')
                subject_key = f"sub-{subject}"
            
            if data is None:
                continue
            
            # Print original trial count for this subject
            print(f"Subject {subject_key} ({dataset_type}): {len(data)} total trials available")
            
            # Detect actual input channels (may be more than original channels due to feature extraction)
            actual_input_channels = data.shape[1]  # (n_samples, n_channels, n_times)
            if actual_input_channels != len(channels):
                print(f"Enhanced preprocessing increased channels from {len(channels)} to {actual_input_channels}")

            # Create data loaders for the current subject
            # Get trial limits from configuration
            from utils import get_trial_limits_from_config
            trial_limits = get_trial_limits_from_config()
            
            # Check if this is sub-001 to get indices for event printing
            if subject_key == 'sub-001' and dataset_type == 'P3':
                train_loader, val_loader, test_loader, train_indices, val_indices, test_indices = create_data_loaders(
                    data, labels, return_indices=True, max_trials_per_split=trial_limits
                )
            else:
                train_loader, val_loader, test_loader = create_data_loaders(data, labels, max_trials_per_split=trial_limits)
            
            # Track trial counts for this subject
            final_key = f"{dataset_type}_{subject_key}" if len(datasets) > 1 else subject_key
            trial_counts[final_key] = {
                'train': len(train_loader.dataset),
                'val': len(val_loader.dataset),
                'test': len(test_loader.dataset)
            }
            
            # Print trial counts for this subject
            print(f"Subject {final_key} trial distribution:")
            print(f"  Train: {len(train_loader.dataset)} trials")
            print(f"  Val:   {len(val_loader.dataset)} trials")
            print(f"  Test:  {len(test_loader.dataset)} trials")
            print(f"  Total: {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)} trials")
            
            print()
            
            
            # Multi-seed training
            subject_accuracies_seed = []
            subject_details_seed = []
            subject_predictions_all = []
            subject_true_labels_all = []
            
            for seed in exp_seeds:
                details, model = run_experiment_with_seed(
                    train_loader, val_loader, test_loader, len(channels), device, seed,
                    exp_classifier, print_model_summary=(i == 0 and seed == exp_seeds[0]),
                    return_details=True, input_channels=actual_input_channels
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
            
            mean_accuracy = np.mean(subject_accuracies_seed)
            all_accuracies[final_key] = mean_accuracy
            
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
            
            # Print results for each subject
            print(f"Subject {final_key}: Acc {all_accuracies[final_key]:.1%}, AUC {avg_auc:.2f}")
            
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
                'accuracy': avg_correct / avg_total if avg_total > 0 else 0,
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
    # Print configuration information
    from config import (
        TRAIN_SIZE, VAL_SIZE, TEST_SIZE, MAX_SUBJECTS_P3, MAX_SUBJECTS_AVO,
        MAX_TRIALS_PER_SUBJECT_TRAIN, MAX_TRIALS_PER_SUBJECT_VAL, MAX_TRIALS_PER_SUBJECT_TEST,
        FIXED_TRIALS_PER_SUBJECT_TRAIN, FIXED_TRIALS_PER_SUBJECT_VAL, FIXED_TRIALS_PER_SUBJECT_TEST
    )
    
    print("="*60)
    print("POOLED TRAINING CONFIGURATION")
    print("="*60)
    print(f"Datasets: {datasets}")
    print(f"Max subjects - P3: {MAX_SUBJECTS_P3}, AVO: {MAX_SUBJECTS_AVO}")
    
    # Check trial limits
    if any(x is not None for x in [FIXED_TRIALS_PER_SUBJECT_TRAIN, FIXED_TRIALS_PER_SUBJECT_VAL, FIXED_TRIALS_PER_SUBJECT_TEST]):
        print("Fixed trial counts per subject:")
        print(f"  Train: {FIXED_TRIALS_PER_SUBJECT_TRAIN}, Val: {FIXED_TRIALS_PER_SUBJECT_VAL}, Test: {FIXED_TRIALS_PER_SUBJECT_TEST}")
        print("Using fixed trial counts instead of ratio-based splitting")
    elif any(x is not None for x in [MAX_TRIALS_PER_SUBJECT_TRAIN, MAX_TRIALS_PER_SUBJECT_VAL, MAX_TRIALS_PER_SUBJECT_TEST]):
        print("Max trial limits per subject:")
        print(f"  Train: {MAX_TRIALS_PER_SUBJECT_TRAIN}, Val: {MAX_TRIALS_PER_SUBJECT_VAL}, Test: {MAX_TRIALS_PER_SUBJECT_TEST}")
        print(f"Train/Val/Test split: {TRAIN_SIZE:.1f}:{VAL_SIZE:.1f}:{TEST_SIZE:.1f}")
    else:
        print("No trial limits - using all available trials per subject")
        print(f"Train/Val/Test split: {TRAIN_SIZE:.1f}:{VAL_SIZE:.1f}:{TEST_SIZE:.1f}")
    print("="*60)
    print()
    
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
    
    # Print summary of all subjects' trial counts
    print(f"\nPooled dataset summary:")
    print(f"  Total subjects: {len(subject_ids)}")
    print(f"  Total trials: {len(all_data)}")
    print(f"  Average trials per subject: {len(all_data) / len(subject_ids):.1f}")
    print()

    # Detect actual input channels for enhanced preprocessing
    actual_input_channels = all_data.shape[1]  # (n_samples, n_channels, n_times)
    if actual_input_channels != len(channels):
        print(f"Enhanced preprocessing increased channels from {len(channels)} to {actual_input_channels}")
    
    # Apply trial limits if specified (for pooled training, this affects the total dataset)
    from utils import get_trial_limits_from_config
    trial_limits = get_trial_limits_from_config()
    
    if trial_limits is not None:
        print("Applying trial limits per subject for pooled training...")
        # For pooled training, we need to apply limits per subject before combining
        # This ensures each subject contributes the specified number of trials
        total_available = len(all_data)
        max_total = None
        
        # Calculate maximum total trials if all limits are specified
        if all(trial_limits.get(split) is not None for split in ['train', 'val', 'test']):
            max_total = sum(trial_limits.values()) * len(subject_ids)  # Per subject limits
        
        print(f"Expected total trials with limits: {max_total} (from {len(subject_ids)} subjects)")
        print(f"Actual total trials available: {total_available}")
        
        # Note: Trial limits will be applied during data splitting in create_data_loaders
        # We don't need to pre-sample here as it would break the per-subject logic
    
    # Create data splits
    if trial_limits is not None and all(trial_limits.get(split) is not None for split in ['train', 'val', 'test']):
        # Use fixed trial counts for pooled training
        max_train = trial_limits['train'] * len(subject_ids)  # Total across all subjects
        max_val = trial_limits['val'] * len(subject_ids)
        max_test = trial_limits['test'] * len(subject_ids)
        
        print(f"Using fixed trial counts for pooled training:")
        print(f"  Train: {max_train} trials ({trial_limits['train']} per subject × {len(subject_ids)} subjects)")
        print(f"  Val: {max_val} trials ({trial_limits['val']} per subject × {len(subject_ids)} subjects)")
        print(f"  Test: {max_test} trials ({trial_limits['test']} per subject × {len(subject_ids)} subjects)")
        
        # Shuffle data first
        indices = np.arange(len(all_data))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        # Split by fixed counts
        train_indices = indices[:max_train]
        val_indices = indices[max_train:max_train + max_val]
        test_indices = indices[max_train + max_val:max_train + max_val + max_test]
        
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)
        
    else:
        # Use ratio-based splitting (original logic)
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
    
    # Since dataset is now balanced at source, no need for weighted sampling  
    train_labels = all_labels[train_indices]
    class_counts = np.bincount(train_labels)
    print(f"Pooled training - Class distribution: {class_counts.tolist()}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Print trial distribution for pooled training
    print(f"Pooled training trial distribution:")
    print(f"  Train: {len(train_loader.dataset)} trials")
    print(f"  Val:   {len(val_loader.dataset)} trials")
    print(f"  Test:  {len(test_loader.dataset)} trials")
    print(f"  Total: {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)} trials")
    print(f"  Subjects: {len(subject_id_to_index)} subjects")
    print()
    
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
                enable_subject_layer=should_use_subject_layer,
                model_name=classifier,
                input_channels=actual_input_channels
            )
            model = model.to(device)
            
            if seed == exp_seeds[0]:
                print(f"\nModel Architecture Summary (Datasets: {datasets})")
                print("="*70)
                print(f"Model type: {type(model).__name__}")
                print(f"Original channels: {len(channels)}, Input channels: {actual_input_channels}")
                print(f"Number of subjects: {n_subjects}")
                print(f"Subject layer enabled: {should_use_subject_layer}")
                print(f"Input shape: (batch_size, {len(channels)}, 128)")
                
                # Count model parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Total parameters: {total_params:,}")
                print(f"Trainable parameters: {trainable_params:,}")
                
                # Print model-specific parameters
                if classifier == 'EEGConformer':
                    from config import (
                        CONFORMER_EMBEDDING_DIM, CONFORMER_NUM_HEADS, CONFORMER_NUM_LAYERS,
                        CONFORMER_CONV_SPATIAL_DIM, CONFORMER_CONV_TEMPORAL_DIM, CONFORMER_ACTIVATION
                    )
                    print(f"\nEEGConformer Configuration:")
                    print(f"  Embedding dim: {CONFORMER_EMBEDDING_DIM}")
                    print(f"  Attention heads: {CONFORMER_NUM_HEADS}")
                    print(f"  Transformer layers: {CONFORMER_NUM_LAYERS}")
                    print(f"  Spatial conv channels: {CONFORMER_CONV_SPATIAL_DIM}")
                    print(f"  Temporal conv channels: {CONFORMER_CONV_TEMPORAL_DIM}")
                    print(f"  Activation: {CONFORMER_ACTIVATION}")
                
                # Print training configuration
                print(f"\nTraining Configuration:")
                print(f"  Learning rate: {LEARNING_RATE}")
                print(f"  Weight decay: {WEIGHT_DECAY}")
                print(f"  Dropout rate: {DROPOUT_RATE}")
                print(f"  Batch size: {BATCH_SIZE}")
                print(f"  Max epochs: {MAX_EPOCHS}")
                print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
                
                # Print data augmentation settings
                print(f"\nData Augmentation:")
                print(f"  Enabled: {USE_DATA_AUGMENTATION}")
                if USE_DATA_AUGMENTATION:
                    print(f"  Noise std: {NOISE_STD}")
                    print(f"  Time shift range: {TIME_SHIFT_RANGE}")
                    print(f"  Label smoothing: {LABEL_SMOOTHING}")
                
                print("="*70 + "\n")
            
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
                        print(f"Warning: Subject {subject_ids[subject_idx]} has only one class in test set: {unique_labels}. Setting AUC to 0.5.")
                        auc = 0.5
                    else:
                        # Check for problematic probability values
                        if np.any(np.isnan(y_proba)) or np.any(np.isinf(y_proba)):
                            print(f"Warning: Subject {subject_ids[subject_idx]} has NaN or infinite values in probabilities. Setting AUC to 0.5.")
                            auc = 0.5
                        else:
                            auc = roc_auc_score(y_subj, y_proba)
                            if np.isnan(auc):
                                print(f"Warning: Subject {subject_ids[subject_idx]} AUC calculation returned NaN. Setting to 0.5.")
                                auc = 0.5
                except Exception as e:
                    print(f"Warning: Subject {subject_ids[subject_idx]} AUC calculation failed: {e}. Setting to 0.5.")
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
                cm = confusion_matrix(y_true, y_pred)
                
                # Handle different confusion matrix shapes
                if cm.size == 1:
                    # Only one class present - all predictions are the same
                    if len(set(y_true)) == 1 and len(set(y_pred)) == 1:
                        # Both true and predicted have same single class
                        tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
                    else:
                        # Different single classes
                        tn, fp, fn, tp = 0, 0, cm[0, 0], 0
                elif cm.size == 4:
                    # Standard 2x2 matrix
                    tn, fp, fn, tp = cm.ravel()
                else:
                    # Unexpected shape - use sklearn functions directly
                    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
                    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
                    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
                    tn = fp = fn = tp = 0
                
                # Calculate precision, recall, f1 from confusion matrix if we have the values
                if cm.size <= 4:
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
                'accuracy': avg_correct / avg_total if avg_total > 0 else 0,
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


# Fusion experiment functions
def run_fusion_experiment(fusion_method: str, domain_adaptation: str = 'none',
                         datasets: List[str] = None, channels: List[str] = None,
                         logger = None):
    """
    运行融合实验

    Args:
        fusion_method: 融合方法 ('graph_gcn', 'none')
        domain_adaptation: 域适应方法 ('ms_mda', 'adversarial', 'none')
        datasets: 数据集列表
        channels: 通道列表
        logger: 日志记录器

    Returns:
        实验结果字典
    """
    from models import create_fusion_model, train_fusion_model, evaluate_fusion_model
    from evaluation_utils import ComprehensiveEvaluator
    from enhanced_preprocessor import FusionDatasetManager

    # 设置默认参数
    if datasets is None:
        datasets = ['P3', 'AVO']
    if channels is None:
        # 融合实验模式：让每个数据集使用自己的完整电极配置
        # None 表示系统将自动为每个数据集使用其原生电极布局
        logger.info("融合模式：使用各数据集的原生电极配置")
        channels = None  # 保持 None，由融合管理器处理

    logger = logger or setup_logger('fusion_experiment', create_file=False)

    try:
        # 初始化融合数据集管理器
        fusion_manager = FusionDatasetManager(
            fusion_method=fusion_method,
            domain_adaptation=domain_adaptation
        )

        # 处理和准备数据
        logger.info(f"Processing datasets: {datasets}")
        processed_datasets = process_fusion_datasets(fusion_manager, datasets, logger)

        # 创建数据加载器
        train_loaders, val_loaders, test_loaders = create_fusion_data_loaders(
            processed_datasets, logger
        )

        # 创建融合模型
        from config import classifier

        # 构建数据集信息，使用真实的电极名称
        from constants import P3_CHANNELS, AVO_CHANNELS

        datasets_info = {}
        for dataset_name in datasets:
            n_channels = processed_datasets[dataset_name]['X_train'].shape[1]

            # 使用真实的电极名称
            if dataset_name == 'P3':
                # P3数据集有增强的通道（原始30通道 + 120个特征 = 150通道）
                # 使用原始电极名称，其余用特征名称
                base_channels = P3_CHANNELS[:min(30, n_channels)]
                if n_channels > 30:
                    # 添加时域特征通道
                    feature_channels = [f'feat_{i}' for i in range(n_channels - len(base_channels))]
                    channels_list = base_channels + feature_channels
                else:
                    channels_list = base_channels[:n_channels]
            elif dataset_name == 'AVO':
                # AVO数据集有增强的通道（原始26通道 + 104个特征 = 130通道）
                base_channels = AVO_CHANNELS[:min(26, n_channels)]
                if n_channels > 26:
                    # 添加时域特征通道
                    feature_channels = [f'feat_{i}' for i in range(n_channels - len(base_channels))]
                    channels_list = base_channels + feature_channels
                else:
                    channels_list = base_channels[:n_channels]
            else:
                # 其他数据集使用通用命名
                channels_list = [f'ch_{i}' for i in range(n_channels)]

            datasets_info[dataset_name] = {
                'channels': channels_list,
                'n_channels': n_channels,
                'n_samples': processed_datasets[dataset_name]['X_train'].shape[2],
                'n_timepoints': processed_datasets[dataset_name]['X_train'].shape[2],
                'n_classes': len(np.unique(processed_datasets[dataset_name]['y_train']))
            }

        model = create_fusion_model(
            model_name=classifier,
            datasets_info=datasets_info,
            fusion_method=fusion_method,
            domain_adaptation=domain_adaptation,
            datasets=datasets,
            input_shape=processed_datasets[datasets[0]]['X_train'].shape[1:],
            logger=logger
        )

        # 预计算各数据集的电极位置张量（与通道顺序一致）
        from fusion_methods import FusionModelFactory
        position_tensors = FusionModelFactory.get_position_tensors(datasets_info)

        # 训练模型
        import torch
        device = get_device()

        logger.info(f"Training fusion model: {fusion_method} with domain adaptation: {domain_adaptation}")
        training_history = train_fusion_model(
            model=model,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            test_loaders=test_loaders,
            device=device,
            fusion_method=fusion_method,
            domain_adaptation=domain_adaptation,
            position_tensors=position_tensors
        )

        # 评估模型（返回平均准确率）
        overall_accuracy = evaluate_fusion_model(
            model=model,
            test_loaders=test_loaders,
            device=device,
            fusion_method=fusion_method,
            domain_adaptation=domain_adaptation,
            position_tensors=position_tensors
        )

        # 标准化评估结果为字典
        evaluation_results = {
            'overall_accuracy': overall_accuracy
        }

        # 综合评估
        evaluator = ComprehensiveEvaluator(logger=logger)
        # 由于当前评估函数未返回逐样本预测，这里暂不进行综合分析
        comprehensive_results = {}

        # 合并结果
        final_results = {
            'fusion_method': fusion_method,
            'domain_adaptation': domain_adaptation,
            'datasets': datasets,
            'training_history': training_history,
            'evaluation_results': evaluation_results,
            'comprehensive_analysis': comprehensive_results,
            'model_params': sum(p.numel() for p in model.parameters()),
        }

        logger.info(f"Fusion experiment completed successfully")
        logger.info(f"Final accuracy: {evaluation_results.get('overall_accuracy', 'N/A')}")

        return final_results

    except Exception as e:
        logger.error(f"Error in fusion experiment: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def process_fusion_datasets(fusion_manager: 'FusionDatasetManager', datasets: List[str], logger):
    """处理融合数据集"""
    processed_datasets = {}

    for dataset_name in datasets:
        logger.info(f"Processing dataset: {dataset_name}")

        if dataset_name == 'P3':
            data_dir = P3_DATA_DIR
        elif dataset_name == 'AVO':
            data_dir = AVO_DATA_DIR
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # 加载和预处理数据
        processed_data = fusion_manager.load_and_prepare_dataset(
            dataset_name=dataset_name,
            data_dir=data_dir
        )

        processed_datasets[dataset_name] = processed_data
        logger.info(f"Dataset {dataset_name} processed: {processed_data['X_train'].shape[0]} training samples")

    return processed_datasets


def create_fusion_data_loaders(processed_datasets: Dict, logger, batch_size: int = None):
    """创建融合数据加载器"""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    batch_size = batch_size or BATCH_SIZE

    train_loaders = {}
    val_loaders = {}
    test_loaders = {}

    for dataset_name, data in processed_datasets.items():
        # 转换为torch张量
        X_train = torch.FloatTensor(data['X_train'])
        y_train = torch.LongTensor(data['y_train'])
        X_val = torch.FloatTensor(data['X_val'])
        y_val = torch.LongTensor(data['y_val'])
        X_test = torch.FloatTensor(data['X_test'])
        y_test = torch.LongTensor(data['y_test'])

        # 创建数据集
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        # 创建数据加载器
        train_loaders[dataset_name] = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loaders[dataset_name] = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        test_loaders[dataset_name] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        logger.info(f"Created data loaders for {dataset_name}")

    return train_loaders, val_loaders, test_loaders


def run_experiment_with_fusion(datasets: List[str], fusion_method: str = 'none',
                              domain_adaptation: str = 'none', channels: List[str] = None,
                              logger = None):
    """
    运行带融合功能的实验

    Args:
        datasets: 数据集列表
        fusion_method: 融合方法
        domain_adaptation: 域适应方法
        channels: 通道列表
        logger: 日志记录器

    Returns:
        实验结果
    """
    # 如果启用融合方法，运行融合实验
    if fusion_method != 'none':
        return run_fusion_experiment(
            fusion_method=fusion_method,
            domain_adaptation=domain_adaptation,
            datasets=datasets,
            channels=channels,
            logger=logger
        )
    else:
        # 否则运行标准实验
        accuracies, trial_counts, prediction_details, _, _, _ = run_experiment(
            datasets=datasets,
            training_mode='pooled',
            channels=channels,
            logger=logger,
            p3_dir=P3_DATA_DIR if 'P3' in datasets else None,
            avo_dir=AVO_DATA_DIR if 'AVO' in datasets else None
        )

        return {
            'fusion_method': fusion_method,
            'domain_adaptation': domain_adaptation,
            'datasets': datasets,
            'accuracies': accuracies,
            'trial_counts': trial_counts,
            'prediction_details': prediction_details
        }
