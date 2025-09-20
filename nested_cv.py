"""
Nested Cross-Validation Framework for EEG Experiments

This module implements nested cross-validation as the primary evaluation method:
- Outer loop: Model comparison and final performance estimation
- Inner loop: Hyperparameter tuning and model selection
- Repeated 10 times for robust statistics
- 95% confidence intervals for all metrics
"""

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import scipy.stats as stats

from config import (
    BATCH_SIZE, MAX_EPOCHS, EARLY_STOPPING_PATIENCE,
    LEARNING_RATE, WEIGHT_DECAY, DROPOUT_RATE,
    seeds
)
from models import create_model, train_model, evaluate
from utils import create_data_loaders
from experiment_logger import setup_logger


class NestedCrossValidation:
    """
    Nested Cross-Validation implementation for robust model evaluation.

    Follows the "politically correct" approach with:
    - Outer CV: 5-fold for final performance estimation
    - Inner CV: 3-fold for hyperparameter tuning
    - Repeated 10 times for statistical significance
    - 95% confidence intervals for all metrics
    """

    def __init__(self,
                 outer_cv_folds: int = 5,
                 inner_cv_folds: int = 3,
                 n_repeats: int = 10,
                 random_state: int = 42,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize nested cross-validation.

        Args:
            outer_cv_folds: Number of folds for outer CV (performance estimation)
            inner_cv_folds: Number of folds for inner CV (hyperparameter tuning)
            n_repeats: Number of times to repeat the entire process
            random_state: Random seed for reproducibility
            logger: Logger instance
        """
        self.outer_cv_folds = outer_cv_folds
        self.inner_cv_folds = inner_cv_folds
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.logger = logger or setup_logger('nested_cv', create_file=False)

        # Define hyperparameter grids for different models
        self.param_grids = {
            'lda': {},  # LDA has no hyperparameters to tune
            'EEGNet': {
                'learning_rate': [0.001, 0.01, 0.05],
                'dropout_rate': [0.2, 0.25, 0.3],
                'weight_decay': [1e-5, 1e-4, 1e-3]
            },
            'SepConv1D': {
                'learning_rate': [0.01, 0.05, 0.1],
                'dropout_rate': [0.2, 0.25, 0.3],
                'weight_decay': [1e-5, 1e-4, 1e-3]
            },
            'EEGConformer': {
                'learning_rate': [0.0001, 0.001, 0.01],
                'dropout_rate': [0.1, 0.2, 0.3],
                'weight_decay': [1e-5, 1e-4, 1e-3]
            },
            'ShallowFBCSPNet': {
                'learning_rate': [0.001, 0.01, 0.05],
                'dropout_rate': [0.2, 0.25, 0.3],
                'weight_decay': [1e-5, 1e-4, 1e-3]
            }
        }

    def run_nested_cv(self,
                     data: np.ndarray,
                     labels: np.ndarray,
                     model_name: str,
                     n_channels: int,
                     device: torch.device,
                     subject_indices: Optional[np.ndarray] = None,
                     **kwargs) -> Dict[str, Any]:
        """
        Run nested cross-validation.

        Args:
            data: Input data (n_samples, n_channels, n_timepoints)
            labels: Target labels (n_samples,)
            model_name: Name of the model to evaluate
            n_channels: Number of input channels
            device: PyTorch device
            subject_indices: Subject indices for subject layer (optional)
            **kwargs: Additional arguments

        Returns:
            Dictionary containing nested CV results
        """
        self.logger.info(f"Starting Nested Cross-Validation for {model_name}")
        self.logger.info(f"Configuration: {self.outer_cv_folds}-fold outer, {self.inner_cv_folds}-fold inner, {self.n_repeats} repeats")

        all_repeat_results = []

        for repeat in range(self.n_repeats):
            self.logger.info(f"Repeat {repeat + 1}/{self.n_repeats}")

            # Create stratified folds for outer CV with different random state each repeat
            outer_cv = StratifiedKFold(
                n_splits=self.outer_cv_folds,
                shuffle=True,
                random_state=self.random_state + repeat
            )

            repeat_scores = []
            fold_results = []

            for fold_idx, (train_val_idx, test_idx) in enumerate(outer_cv.split(data, labels)):
                self.logger.info(f"  Outer fold {fold_idx + 1}/{self.outer_cv_folds}")

                # Split data
                X_train_val, X_test = data[train_val_idx], data[test_idx]
                y_train_val, y_test = labels[train_val_idx], labels[test_idx]

                if subject_indices is not None:
                    subj_train_val = subject_indices[train_val_idx]
                    subj_test = subject_indices[test_idx]
                else:
                    subj_train_val = None
                    subj_test = None

                # Inner CV for hyperparameter tuning
                best_params = self._inner_cv_hyperparameter_tuning(
                    X_train_val, y_train_val, model_name, n_channels, device,
                    subj_train_val, fold_idx, repeat
                )

                # Train final model with best parameters on full train_val set
                final_model = self._train_final_model(
                    X_train_val, y_train_val, model_name, n_channels, device,
                    best_params, subj_train_val
                )

                # Evaluate on test set
                test_metrics = self._evaluate_model(
                    final_model, X_test, y_test, model_name, device, subj_test
                )

                repeat_scores.append(test_metrics['accuracy'])
                fold_results.append({
                    'fold': fold_idx,
                    'best_params': best_params,
                    'test_metrics': test_metrics,
                    'test_size': len(X_test)
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

        self.logger.info(f"Nested CV completed for {model_name}")
        self.logger.info(f"Final accuracy: {final_results['mean_accuracy']:.4f} ± {final_results['std_accuracy']:.4f}")
        self.logger.info(f"95% CI: [{final_results['ci_lower']:.4f}, {final_results['ci_upper']:.4f}]")

        return final_results

    def _inner_cv_hyperparameter_tuning(self,
                                       X_train_val: np.ndarray,
                                       y_train_val: np.ndarray,
                                       model_name: str,
                                       n_channels: int,
                                       device: torch.device,
                                       subject_indices: Optional[np.ndarray],
                                       outer_fold: int,
                                       repeat: int) -> Dict[str, Any]:
        """
        Inner CV loop for hyperparameter tuning.
        """
        if model_name.lower() == 'lda':
            # LDA has no hyperparameters to tune
            return {}

        param_grid = self.param_grids.get(model_name, self.param_grids['EEGNet'])

        if not param_grid:
            # Return default parameters if no grid defined
            return {
                'learning_rate': LEARNING_RATE,
                'dropout_rate': DROPOUT_RATE,
                'weight_decay': WEIGHT_DECAY
            }

        # Create parameter combinations
        param_combinations = list(ParameterGrid(param_grid))

        # Inner CV
        inner_cv = StratifiedKFold(
            n_splits=self.inner_cv_folds,
            shuffle=True,
            random_state=self.random_state + repeat * 100 + outer_fold
        )

        best_score = -np.inf
        best_params = None

        for param_idx, params in enumerate(param_combinations):
            inner_scores = []

            for inner_fold, (train_idx, val_idx) in enumerate(inner_cv.split(X_train_val, y_train_val)):
                X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
                y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

                if subject_indices is not None:
                    subj_train = subject_indices[train_idx]
                    subj_val = subject_indices[val_idx]
                else:
                    subj_train = None
                    subj_val = None

                # Train model with current parameters
                model = self._train_model_with_params(
                    X_train, y_train, X_val, y_val, model_name, n_channels,
                    device, params, subj_train, subj_val
                )

                # Evaluate on validation set
                val_metrics = self._evaluate_model(model, X_val, y_val, model_name, device, subj_val)
                inner_scores.append(val_metrics['accuracy'])

            # Calculate mean validation score for this parameter combination
            mean_score = np.mean(inner_scores)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params.copy()

        # Add default values for any missing parameters
        if best_params is None:
            best_params = {}

        best_params.setdefault('learning_rate', LEARNING_RATE)
        best_params.setdefault('dropout_rate', DROPOUT_RATE)
        best_params.setdefault('weight_decay', WEIGHT_DECAY)

        return best_params

    def _train_model_with_params(self,
                               X_train: np.ndarray,
                               y_train: np.ndarray,
                               X_val: np.ndarray,
                               y_val: np.ndarray,
                               model_name: str,
                               n_channels: int,
                               device: torch.device,
                               params: Dict[str, Any],
                               subj_train: Optional[np.ndarray] = None,
                               subj_val: Optional[np.ndarray] = None) -> Any:
        """
        Train a model with specific hyperparameters.
        """
        if model_name.lower() == 'lda':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            model = LinearDiscriminantAnalysis()
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            model.fit(X_train_flat, y_train)
            return model

        # Create data loaders
        # Since create_data_loaders doesn't accept separate validation data,
        # we need to create them manually for hyperparameter tuning
        from torch.utils.data import DataLoader, TensorDataset

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Create model
        actual_input_channels = X_train.shape[1]
        model = create_model(
            n_channels,
            is_lda=False,
            model_name=model_name,
            input_channels=actual_input_channels
        )
        model = model.to(device)

        # Temporarily override global parameters with tuned parameters
        original_lr = LEARNING_RATE
        original_wd = WEIGHT_DECAY
        original_dr = DROPOUT_RATE

        # Apply hyperparameters to model configuration
        import config
        config.LEARNING_RATE = params.get('learning_rate', LEARNING_RATE)
        config.WEIGHT_DECAY = params.get('weight_decay', WEIGHT_DECAY)
        config.DROPOUT_RATE = params.get('dropout_rate', DROPOUT_RATE)

        try:
            # Create a dummy test loader for training function
            test_loader = val_loader  # Use validation as test for inner CV

            # Train model with reduced epochs for efficiency
            original_epochs = MAX_EPOCHS
            original_patience = EARLY_STOPPING_PATIENCE
            config.MAX_EPOCHS = min(10, MAX_EPOCHS)  # Very limited for testing
            config.EARLY_STOPPING_PATIENCE = min(5, EARLY_STOPPING_PATIENCE)

            train_model(model, train_loader, val_loader, test_loader, device, is_lda=False)

            # Restore original configuration
            config.MAX_EPOCHS = original_epochs
            config.EARLY_STOPPING_PATIENCE = original_patience

        finally:
            # Restore original parameters
            config.LEARNING_RATE = original_lr
            config.WEIGHT_DECAY = original_wd
            config.DROPOUT_RATE = original_dr

        return model

    def _train_final_model(self,
                          X_train_val: np.ndarray,
                          y_train_val: np.ndarray,
                          model_name: str,
                          n_channels: int,
                          device: torch.device,
                          best_params: Dict[str, Any],
                          subject_indices: Optional[np.ndarray] = None) -> Any:
        """
        Train the final model with best hyperparameters on full train+val set.
        """
        if model_name.lower() == 'lda':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            model = LinearDiscriminantAnalysis()
            X_train_flat = X_train_val.reshape(X_train_val.shape[0], -1)
            model.fit(X_train_flat, y_train_val)
            return model

        # Split train_val into train and val for final training
        val_size = 0.2  # Use 20% for validation
        from sklearn.model_selection import train_test_split

        train_idx, val_idx = train_test_split(
            range(len(X_train_val)),
            test_size=val_size,
            stratify=y_train_val,
            random_state=self.random_state
        )

        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

        # Create data loaders
        # Since create_data_loaders doesn't accept separate validation data,
        # we need to create them manually for hyperparameter tuning
        from torch.utils.data import DataLoader, TensorDataset

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Create model
        actual_input_channels = X_train.shape[1]
        model = create_model(
            n_channels,
            is_lda=False,
            model_name=model_name,
            input_channels=actual_input_channels
        )
        model = model.to(device)

        # Apply best hyperparameters
        import config
        original_lr = config.LEARNING_RATE
        original_wd = config.WEIGHT_DECAY
        original_dr = config.DROPOUT_RATE

        config.LEARNING_RATE = best_params.get('learning_rate', LEARNING_RATE)
        config.WEIGHT_DECAY = best_params.get('weight_decay', WEIGHT_DECAY)
        config.DROPOUT_RATE = best_params.get('dropout_rate', DROPOUT_RATE)

        try:
            # Use validation loader as test for training function
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
            'nested_cv_config': {
                'outer_folds': self.outer_cv_folds,
                'inner_folds': self.inner_cv_folds,
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
    Convenience function to run nested cross-validation experiment.

    Args:
        data: Input data (n_samples, n_channels, n_timepoints)
        labels: Target labels (n_samples,)
        model_name: Name of the model to evaluate
        n_channels: Number of input channels
        device: PyTorch device
        logger: Logger instance
        **kwargs: Additional arguments for NestedCrossValidation

    Returns:
        Nested CV results with 95% confidence intervals
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
        model_name='EEGNet',
        n_channels=n_channels,
        device=device,
        logger=logger,
        n_repeats=2  # Reduced for testing
    )

    print(f"Results: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")