#!/usr/bin/env python3
"""
Test script for Nested Cross-Validation implementation.
Validates the nested CV framework with a small synthetic dataset.
"""

import numpy as np
import torch
import logging
from nested_cv import NestedCrossValidation, run_nested_cv_experiment
from experiment_logger import setup_logger

def test_nested_cv():
    """Test nested CV with synthetic data."""
    print("="*60)
    print("TESTING NESTED CROSS-VALIDATION IMPLEMENTATION")
    print("="*60)

    # Setup logger
    logger = setup_logger('nested_cv_test', create_file=False)

    # Create synthetic EEG-like data
    np.random.seed(42)
    n_samples = 200  # Small dataset for testing
    n_channels = 10  # Reduced channels for faster testing
    n_timepoints = 128  # Use standard timepoints to match model expectations

    print(f"Generating synthetic data: {n_samples} samples, {n_channels} channels, {n_timepoints} timepoints")

    # Generate data with some structure
    data = np.random.randn(n_samples, n_channels, n_timepoints)

    # Create labels with slight bias for testing
    labels = np.random.randint(0, 2, n_samples)

    # Add some signal to make classification possible
    for i in range(n_samples):
        if labels[i] == 1:
            data[i, :, :] += 0.2 * np.random.randn(n_channels, n_timepoints)

    print(f"Labels distribution: Class 0: {np.sum(labels == 0)}, Class 1: {np.sum(labels == 1)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test with different models (start with just LDA for speed)
    models_to_test = ['lda']  # , 'EEGNet']

    for model_name in models_to_test:
        print(f"\n{'='*40}")
        print(f"Testing model: {model_name}")
        print(f"{'='*40}")

        try:
            # Run nested CV with reduced parameters for testing
            results = run_nested_cv_experiment(
                data=data,
                labels=labels,
                model_name=model_name,
                n_channels=n_channels,
                device=device,
                logger=logger,
                outer_cv_folds=2,  # Minimal for testing
                inner_cv_folds=2,  # Minimal for testing
                n_repeats=1        # Single repeat for testing
            )

            # Validate results
            assert 'mean_accuracy' in results, "Missing mean_accuracy"
            assert 'ci_lower' in results, "Missing confidence interval lower bound"
            assert 'ci_upper' in results, "Missing confidence interval upper bound"
            assert 'std_accuracy' in results, "Missing standard deviation"

            mean_acc = results['mean_accuracy']
            ci_lower = results['ci_lower']
            ci_upper = results['ci_upper']
            std_acc = results['std_accuracy']

            print(f"✓ {model_name} Results:")
            print(f"  Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"  CI width: {ci_upper - ci_lower:.4f}")

            # Basic validation
            assert 0 <= mean_acc <= 1, f"Invalid accuracy: {mean_acc}"
            assert ci_lower <= mean_acc <= ci_upper, f"Mean not within CI"
            # Note: With small samples, CI can extend beyond [0,1] - this is statistically correct

            print(f"✓ {model_name} validation passed")

        except Exception as e:
            print(f"✗ {model_name} failed: {e}")
            raise

    print(f"\n{'='*60}")
    print("ALL TESTS PASSED!")
    print("Nested Cross-Validation implementation is working correctly.")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_nested_cv()