#!/usr/bin/env python3
"""
Quick start script for TF-DWT hyperparameter tuning

This script provides pre-configured tuning setups for different scenarios:
1. Quick test (5 trials) - for testing the system
2. Standard tuning (50 trials) - balanced exploration
3. Extensive tuning (200 trials) - thorough search
4. Focus on specific parameter groups

Usage examples:
    python run_tuning_example.py --mode quick
    python run_tuning_example.py --mode standard
    python run_tuning_example.py --mode extensive
    python run_tuning_example.py --mode custom --trials 100
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_quick_test():
    """Run a quick 5-trial test to verify the system works."""
    print("STARTING QUICK TEST (5 trials)")
    print("Estimated time: 1.5-3 hours")
    print("Monitor progress: Each trial will show detailed progress")
    print("Results will be saved to: quick_test_results/")
    print("=" * 60)

    cmd = [
        sys.executable, "tune_tfdwt.py",
        "--strategy", "random",
        "--n_trials", "5",
        "--results_dir", "quick_test_results"
    ]
    return subprocess.run(cmd)


def run_standard_tuning():
    """Run standard 50-trial random search."""
    print("STARTING STANDARD TUNING (50 trials)")
    print("Estimated time: 12-25 hours")
    print("Monitor progress: Each trial will show detailed progress")
    print("Results will be saved to: standard_tuning_results/")
    print("You can interrupt and resume anytime")
    print("=" * 60)

    cmd = [
        sys.executable, "tune_tfdwt.py",
        "--strategy", "random",
        "--n_trials", "50",
        "--results_dir", "standard_tuning_results"
    ]
    return subprocess.run(cmd)


def run_extensive_tuning():
    """Run extensive 200-trial random search."""
    print("Running extensive tuning (200 trials)...")
    cmd = [
        sys.executable, "tune_tfdwt.py",
        "--strategy", "random",
        "--n_trials", "200",
        "--results_dir", "extensive_tuning_results"
    ]
    return subprocess.run(cmd)


def run_grid_search():
    """Run grid search over key parameters."""
    print("Running grid search...")
    cmd = [
        sys.executable, "tune_tfdwt.py",
        "--strategy", "grid",
        "--results_dir", "grid_search_results"
    ]
    return subprocess.run(cmd)


def run_custom_tuning(n_trials):
    """Run custom number of trials."""
    print(f"Running custom tuning ({n_trials} trials)...")
    cmd = [
        sys.executable, "tune_tfdwt.py",
        "--strategy", "random",
        "--n_trials", str(n_trials),
        "--results_dir", f"custom_tuning_{n_trials}_results"
    ]
    return subprocess.run(cmd)


def create_focused_tuning_config():
    """Create a specialized tuning config focusing on most important parameters."""
    focused_tuner_code = '''#!/usr/bin/env python3
"""
Focused hyperparameter tuning - concentrates on the most impactful parameters.
Based on EEG classification literature, these parameters typically have the most impact:
1. Learning rate and weight decay (optimization)
2. Model architecture choice
3. TF-DWT domain weighting parameters
"""

import sys
import os
sys.path.append('.')

from tune_tfdwt import TFDWTTuner
import numpy as np

class FocusedTFDWTTuner(TFDWTTuner):
    def get_parameter_space(self):
        """Focused parameter space on most impactful parameters."""
        return {
            # Most critical: optimization parameters
            'LEARNING_RATE': {
                'type': 'log_uniform',
                'low': 0.001,
                'high': 0.1,
                'default': 0.01
            },
            'WEIGHT_DECAY': {
                'type': 'log_uniform',
                'low': 1e-5,
                'high': 1e-2,
                'default': 1e-4
            },

            # Second most critical: model choice
            'classifier': {
                'type': 'choice',
                'values': ['EEGConformer', 'SepConv1DLite', 'EEGNetv4'],
                'default': 'EEGConformer'
            },

            # TF-DWT specific: domain balance
            'w_small_clip_max': {
                'type': 'uniform',
                'low': 2.0,
                'high': 8.0,
                'default': 6.0
            },
            'lambda_mmd_base': {
                'type': 'uniform',
                'low': 0.05,
                'high': 0.3,
                'default': 0.1
            },

            # Training stability
            'DROPOUT_RATE': {
                'type': 'uniform',
                'low': 0.1,
                'high': 0.4,
                'default': 0.25
            },
            'BATCH_SIZE': {
                'type': 'choice',
                'values': [24, 32, 48],
                'default': 32
            },

            # Keep other parameters at defaults
            'MAX_EPOCHS': {'type': 'fixed', 'value': 500},
            'EARLY_STOPPING_PATIENCE': {'type': 'fixed', 'value': 50},
            'NOISE_STD': {'type': 'fixed', 'value': 0.005},
            'TIME_SHIFT_RANGE': {'type': 'fixed', 'value': 5},
            'LABEL_SMOOTHING': {'type': 'fixed', 'value': 0.05},
            'gradient_clip_norm': {'type': 'fixed', 'value': 5.0},
            'warmup_ratio': {'type': 'fixed', 'value': 0.1},
        }

    def _random_sample(self, param_space, n_trials):
        """Enhanced sampling for focused parameters."""
        samples = []
        for _ in range(n_trials):
            sample = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'fixed':
                    sample[param_name] = param_config['value']
                elif param_config['type'] == 'uniform':
                    sample[param_name] = np.random.uniform(param_config['low'], param_config['high'])
                elif param_config['type'] == 'log_uniform':
                    sample[param_name] = np.exp(np.random.uniform(
                        np.log(param_config['low']), np.log(param_config['high'])
                    ))
                elif param_config['type'] == 'choice':
                    sample[param_name] = np.random.choice(param_config['values'])
                else:
                    sample[param_name] = param_config['default']
            samples.append(sample)
        return samples

if __name__ == "__main__":
    tuner = FocusedTFDWTTuner("config.py", "focused_tuning_results")
    results = tuner.run_tuning("random", 75)  # 75 focused trials

    # Generate report
    report = tuner.generate_report()
    with open("focused_tuning_results/focused_report.md", "w") as f:
        f.write(report)

    print(f"\\nFocused tuning completed!")
    print(f"Best accuracy: {results['best_score']:.4f}")
    print("Results in: focused_tuning_results/")
'''

    with open("run_focused_tuning.py", "w") as f:
        f.write(focused_tuner_code)

    print("Created focused tuning script: run_focused_tuning.py")


def print_usage():
    """Print detailed usage information."""
    print("""
TF-DWT Hyperparameter Tuning - Usage Guide
==========================================

This tool helps you find the best hyperparameters for your TF-DWT model with:
- NESTED_CV_TRIALS_PER_SUBJECT_P3 = 20
- NESTED_CV_TRIALS_PER_SUBJECT_AVO = 200

Available modes:
1. quick    - 5 trials, ~30min, test if system works
2. standard - 50 trials, ~5-8 hours, good balance
3. extensive- 200 trials, ~20-30 hours, thorough search
4. grid     - Grid search, ~3-6 hours, systematic exploration
5. focused  - 75 trials on key params, ~8-12 hours, targeted search
6. custom   - Specify number of trials

Examples:
  python run_tuning_example.py --mode quick
  python run_tuning_example.py --mode standard
  python run_tuning_example.py --mode focused
  python run_tuning_example.py --mode custom --trials 100

Tunable Parameters:
==================
Core Training:
- LEARNING_RATE: 1e-4 to 1e-1 (log scale)
- WEIGHT_DECAY: 1e-6 to 1e-2 (log scale)
- DROPOUT_RATE: 0.1 to 0.5
- BATCH_SIZE: [16, 24, 32, 48, 64]
- MAX_EPOCHS: 200 to 800

Model Architecture:
- classifier: [EEGConformer, EEGNetv4, ShallowFBCSPNet, SepConv1DLite, EEGChannelNet]

TF-DWT Specific:
- w_small_clip_max: 2.0 to 8.0 (domain weight clipping)
- lambda_mmd_base: 0.05 to 0.5 (MMD alignment strength)
- gradient_clip_norm: 1.0 to 10.0
- warmup_ratio: 0.05 to 0.25

Data Augmentation:
- NOISE_STD: 0.001 to 0.02
- TIME_SHIFT_RANGE: 2 to 15 samples
- LABEL_SMOOTHING: 0.0 to 0.2

Results will be saved in the specified results directory with:
- tuning_results.json: All trial results
- best_config.py: Best parameter configuration
- tuning_report.md: Analysis report
""")


def main():
    parser = argparse.ArgumentParser(description='TF-DWT Hyperparameter Tuning Runner')
    parser.add_argument('--mode',
                        choices=['quick', 'standard', 'extensive', 'grid', 'focused', 'custom', 'help'],
                        default='help',
                        help='Tuning mode to run')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of trials for custom mode')

    args = parser.parse_args()

    if args.mode == 'help':
        print_usage()
        return

    # Verify main files exist
    if not Path("main_tfdwt.py").exists():
        print("ERROR: main_tfdwt.py not found in current directory")
        return 1

    if not Path("config.py").exists():
        print("ERROR: config.py not found in current directory")
        return 1

    if not Path("tune_tfdwt.py").exists():
        print("ERROR: tune_tfdwt.py not found. Please run the tuning script creation first.")
        return 1

    # Run selected mode
    result = None
    if args.mode == 'quick':
        result = run_quick_test()
    elif args.mode == 'standard':
        result = run_standard_tuning()
    elif args.mode == 'extensive':
        result = run_extensive_tuning()
    elif args.mode == 'grid':
        result = run_grid_search()
    elif args.mode == 'focused':
        create_focused_tuning_config()
        result = subprocess.run([sys.executable, "run_focused_tuning.py"])
    elif args.mode == 'custom':
        result = run_custom_tuning(args.trials)

    if result:
        return result.returncode
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)