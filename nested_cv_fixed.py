"""
Nested Cross-Validation Framework for EEG Experiments (FIXED VERSION)

This module implements cross-validation without hyperparameter tuning:
- Outer loop: Model performance estimation with proper data splits
- Repeated with different random seeds for robust statistics  
- 95% confidence intervals for all metrics
- FIXED: No data leakage - proper train/val/test splits
"""

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import scipy.stats as stats

from config import (
    BATCH_SIZE, MAX_EPOCHS, EARLY_STOPPING_PATIENCE,
    LEARNING_RATE, WEIGHT_DECAY, DROPOUT_RATE,
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE,  # Import size configurations
    seeds
)
from models import create_model, train_model, evaluate
from utils import create_data_loaders
from experiment_logger import setup_logger


class NestedCrossValidation:
    """
    Cross-Validation implementation for robust model evaluation.

    Simplified approach with:
    - Cross-Validation: 5-fold for performance estimation
    - Repeated 10 times for statistical significance
    - 95% confidence intervals for all metrics
    - Uses default hyperparameters (no tuning)
    - FIXED: Proper data splits to prevent leakage
    """

    def __init__(self,
                 outer_cv_folds: int = 5,
                 n_repeats: int = 10,
                 random_state: int = 42,
                 seeds: Optional[List[int]] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize cross-validation.

        Args:
            outer_cv_folds: Number of folds for CV (performance estimation)
            n_repeats: Number of times to repeat the entire process
            random_state: Random seed for reproducibility
            seeds: List of random seeds to use for each repeat
            logger: Logger instance
        """
        self.outer_cv_folds = outer_cv_folds
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.seeds = seeds or [42, 123, 456, 789, 321]  # Default seeds if not provided
        self.logger = logger or setup_logger('nested_cv', create_file=False)

    def run_nested_cv(self,
                     data: np.ndarray,
                     labels: np.ndarray,
                     model_name: str,
                     n_channels: int,
                     device: torch.device,
                     subject_indices: Optional[np.ndarray] = None,
                     **kwargs) -> Dict[str, Any]:
        """
        Run cross-validation without hyperparameter tuning (fixed version).

        Args:
            data: Input data (n_samples, n_channels, n_timepoints)
            labels: Target labels (n_samples,)
            model_name: Name of the model to evaluate
            n_channels: Number of input channels
            device: PyTorch device
            subject_indices: Subject indices for subject layer (optional)
            **kwargs: Additional arguments

        Returns:
            Dictionary containing CV results with 95% confidence intervals
        """
        self.logger.info(f"Starting Cross-Validation for {model_name}")
        self.logger.info(f"Configuration: {self.outer_cv_folds}-fold CV, {self.n_repeats} repeats")
        self.logger.info(f"Data split ratios: Train={TRAIN_SIZE}, Val={VAL_SIZE}, Test={TEST_SIZE}")

        all_repeat_results = []

        for repeat in range(self.n_repeats):
            self.logger.info(f"Repeat {repeat + 1}/{self.n_repeats}")

            # Create stratified folds for CV with different random state each repeat
            cv = StratifiedKFold(
                n_splits=self.outer_cv_folds,
                shuffle=True,
                random_state=self.seeds[repeat % len(self.seeds)]
            )

            repeat_scores = []
            fold_results = []

            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(data, labels)):
                self.logger.info(f"  Fold {fold_idx + 1}/{self.outer_cv_folds}")

                # Split data - test_idx is the TRUE test set for this fold
                X_train_fold, X_test_fold = data[train_idx], data[test_idx]
                y_train_fold, y_test_fold = labels[train_idx], labels[test_idx]

                if subject_indices is not None:
                    subj_train_fold = subject_indices[train_idx]
                    subj_test_fold = subject_indices[test_idx]
                else:
                    subj_train_fold = None
                    subj_test_fold = None

                # Train model with proper train/val split (FIXED)
                model = self._train_model_with_proper_split(
                    X_train_fold, y_train_fold, model_name, n_channels, device, subj_train_fold
                )

                # Evaluate on the TRUE test set (never seen during training)
                test_metrics = self._evaluate_model(
                    model, X_test_fold, y_test_fold, model_name, device, subj_test_fold
                )

                repeat_scores.append(test_metrics['accuracy'])
                fold_results.append({
                    'fold': fold_idx,
                    'test_metrics': test_metrics,
                    'test_size': len(X_test_fold)
                })

                self.logger.info(f"    Test accuracy: {test_metrics['accuracy']:.4f}")

            # Store results for this repeat
            repeat_result = {
                'repeat': repeat,
                'fold_results': fold_results,
                'mean_accuracy': np.mean(repeat_scores),
                'std_accuracy': np.std(repeat_scores),
                'scores': repeat_scores
            }
            all_repeat_results.append(repeat_result)

            self.logger.info(f"  Repeat {repeat + 1} mean accuracy: {np.mean(repeat_scores):.4f} ± {np.std(repeat_scores):.4f}")

        # Compile final results with confidence intervals
        final_results = self._compile_final_results(all_repeat_results, model_name)

        self.logger.info(f"Cross-Validation completed for {model_name}")
        self.logger.info(f"Final accuracy: {final_results['mean_accuracy']:.4f} ± {final_results['std_accuracy']:.4f}")
        self.logger.info(f"95% CI: [{final_results['ci_lower']:.4f}, {final_results['ci_upper']:.4f}]")

        return final_results

    def _train_model_with_proper_split(self,
                                     X_train_fold: np.ndarray,
                                     y_train_fold: np.ndarray,
                                     model_name: str,
                                     n_channels: int,
                                     device: torch.device,
                                     subject_indices: Optional[np.ndarray] = None) -> Any:
        """
        Train a model with proper train/validation split (FIXED - No data leakage).
        """
        if model_name.lower() == 'lda':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            model = LinearDiscriminantAnalysis()
            X_train_flat = X_train_fold.reshape(X_train_fold.shape[0], -1)
            model.fit(X_train_flat, y_train_fold)
            return model

        # FIXED: Properly split the fold's training data into train and validation
        # Calculate the ratio for train/val split within the fold
        train_val_total = TRAIN_SIZE + VAL_SIZE  # 0.7 + 0.1 = 0.8
        train_ratio_within_fold = TRAIN_SIZE / train_val_total  # 0.7 / 0.8 = 0.875

        # Split fold training data into actual train and validation sets
        X_train_actual, X_val, y_train_actual, y_val = train_test_split(
            X_train_fold, y_train_fold,
            train_size=train_ratio_within_fold,
            stratify=y_train_fold,
            random_state=self.random_state
        )

        self.logger.info(f"    Fold split: Train={len(X_train_actual)}, Val={len(X_val)}")

        # Create proper data loaders with NO leakage
        from torch.utils.data import DataLoader, TensorDataset

        train_dataset = TensorDataset(torch.FloatTensor(X_train_actual), torch.LongTensor(y_train_actual))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Create model
        actual_input_channels = X_train_fold.shape[1]
        model = create_model(
            n_channels,
            is_lda=False,
            model_name=model_name,
            input_channels=actual_input_channels
        )
        model = model.to(device)

        # Apply default hyperparameters
        import config
        original_lr = config.LEARNING_RATE
        original_wd = config.WEIGHT_DECAY
        original_dr = config.DROPOUT_RATE

        config.LEARNING_RATE = LEARNING_RATE
        config.WEIGHT_DECAY = WEIGHT_DECAY
        config.DROPOUT_RATE = DROPOUT_RATE

        try:
            # FIXED: Use proper validation set, not training data
            # For the test_loader parameter, we use val_loader to avoid any test data leakage
            train_model(model, train_loader, val_loader, val_loader, device, is_lda=False)
        finally:
            # Restore original parameters
            config.LEARNING_RATE = original_lr
            config.WEIGHT_DECAY = original_wd
            config.DROPOUT_RATE = original_dr

        return model

    def _evaluate_model(self,
                       model: Any,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       model_name: str,
                       device: torch.device,
                       subject_indices: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate model and return comprehensive metrics.
        """
        if model_name.lower() == 'lda':
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            predictions = model.predict(X_test_flat)
            try:
                probabilities = model.predict_proba(X_test_flat)[:, 1]
            except:
                probabilities = predictions.astype(float)
        else:
            # Neural network evaluation
            from torch.utils.data import DataLoader, TensorDataset
            test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            model.eval()
            predictions = []
            probabilities = []

            with torch.no_grad():
                for batch in test_loader:
                    if len(batch) == 3:
                        x, y, _ = batch
                    else:
                        x, y = batch

                    from models import normalize_data
                    x = normalize_data(x).to(device)
                    scores = model(x)

                    if scores.ndim > 2:
                        scores = scores.view(scores.size(0), -1)

                    probs = torch.softmax(scores, dim=1)
                    _, pred = scores.max(1)

                    predictions.extend(pred.cpu().numpy())
                    probabilities.extend(probs[:, 1].cpu().numpy() if probs.shape[1] > 1 else pred.cpu().numpy())

            predictions = np.array(predictions)
            probabilities = np.array(probabilities)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='binary', zero_division=0)
        recall = recall_score(y_test, predictions, average='binary', zero_division=0)
        f1 = f1_score(y_test, predictions, average='binary', zero_division=0)

        try:
            if len(np.unique(y_test)) > 1:
                auc = roc_auc_score(y_test, probabilities)
            else:
                auc = 0.5
        except:
            auc = 0.5

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'n_samples': len(y_test)
        }

    def _compile_final_results(self, all_repeat_results: List[Dict], model_name: str) -> Dict[str, Any]:
        """
        Compile final results with confidence intervals.
        """
        # Extract all accuracy scores across repeats
        all_accuracies = []
        all_metrics = {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'auc': []
        }

        for repeat_result in all_repeat_results:
            all_accuracies.extend(repeat_result['scores'])

            # Collect other metrics from fold results
            for fold_result in repeat_result['fold_results']:
                metrics = fold_result['test_metrics']
                for metric_name in all_metrics.keys():
                    all_metrics[metric_name].append(metrics[metric_name])

        # Calculate statistics for accuracy
        mean_acc = np.mean(all_accuracies)
        std_acc = np.std(all_accuracies, ddof=1)  # Sample standard deviation

        # Calculate 95% confidence interval
        n_samples = len(all_accuracies)
        t_critical = stats.t.ppf(0.975, df=n_samples-1)  # 97.5th percentile for 95% CI
        margin_of_error = t_critical * (std_acc / np.sqrt(n_samples))

        ci_lower = mean_acc - margin_of_error
        ci_upper = mean_acc + margin_of_error

        # Calculate statistics for other metrics
        other_stats = {}
        for metric_name, values in all_metrics.items():
            if values:  # Check if we have values
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)
                margin = t_critical * (std_val / np.sqrt(len(values)))

                other_stats[metric_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'ci_lower': mean_val - margin,
                    'ci_upper': mean_val + margin
                }

        # Compile final results
        final_results = {
            'model_name': model_name,
            'cv_config': {
                'cv_folds': self.outer_cv_folds,
                'n_repeats': self.n_repeats,
                'total_evaluations': len(all_accuracies)
            },
            'accuracy': {
                'mean': mean_acc,
                'std': std_acc,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'margin_of_error': margin_of_error
            },
            'mean_accuracy': mean_acc,  # For backward compatibility
            'std_accuracy': std_acc,    # For backward compatibility
            'ci_lower': ci_lower,       # For backward compatibility
            'ci_upper': ci_upper,       # For backward compatibility
            'other_metrics': other_stats,
            'all_repeat_results': all_repeat_results,
            'raw_scores': all_accuracies,
            'statistical_significance': {
                'n_samples': n_samples,
                't_critical': t_critical,
                'degrees_of_freedom': n_samples - 1,
                'confidence_level': 0.95
            }
        }

        return final_results


def run_nested_cv_experiment(data: np.ndarray,
                            labels: np.ndarray,
                            model_name: str,
                            n_channels: int,
                            device: torch.device,
                            logger: Optional[logging.Logger] = None,
                            **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run cross-validation experiment (FIXED VERSION).

    Args:
        data: Input data (n_samples, n_channels, n_timepoints)
        labels: Target labels (n_samples,)
        model_name: Name of the model to evaluate
        n_channels: Number of input channels
        device: PyTorch device
        logger: Logger instance
        **kwargs: Additional arguments for NestedCrossValidation

    Returns:
        Cross-validation results with 95% confidence intervals
    """
    cv = NestedCrossValidation(logger=logger, **kwargs)
    return cv.run_nested_cv(data, labels, model_name, n_channels, device)


if __name__ == "__main__":
    # Example usage
    logger = setup_logger('nested_cv_test')

    # Generate dummy data for testing
    np.random.seed(42)
    n_samples, n_channels, n_timepoints = 200, 30, 128
    data = np.random.randn(n_samples, n_channels, n_timepoints)
    labels = np.random.randint(0, 2, n_samples)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Run nested CV
    results = run_nested_cv_experiment(
        data=data,
        labels=labels,
        model_name='lda',
        n_channels=n_channels,
        device=device,
        logger=logger,
        n_repeats=2  # Reduced for testing
    )

    print(f"Results: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
