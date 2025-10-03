#!/usr/bin/env python
"""
Automated optimization script for TF-DWT parameter tuning
Goal:
- P3=80, AVO=10 -> AVO accuracy >= 0.66 (5 consecutive runs)
- P3=10, AVO=80 -> P3 accuracy >= 0.62 (5 consecutive runs)
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# Configuration scenarios to test
SCENARIOS = {
    'avo_focused': {
        'NESTED_CV_TRIALS_PER_SUBJECT_P3': 80,
        'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 10,
        'target_metric': 'avo_accuracy',
        'target_value': 0.66
    },
    'p3_focused': {
        'NESTED_CV_TRIALS_PER_SUBJECT_P3': 10,
        'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 80,
        'target_metric': 'p3_accuracy',
        'target_value': 0.62
    }
}

# Hyperparameter search space (progressive from small to large changes)
PARAM_CONFIGS = [
    # Baseline
    {
        'name': 'baseline',
        'LEARNING_RATE': 0.01,
        'BATCH_SIZE': 128,
        'WEIGHT_DECAY': 1e-4,
        'DROPOUT_RATE': 0.25,
        'MAX_EPOCHS': 500,
        'EARLY_STOPPING_PATIENCE': 50
    },
    # Smaller learning rate
    {
        'name': 'lower_lr',
        'LEARNING_RATE': 0.005,
        'BATCH_SIZE': 128,
        'WEIGHT_DECAY': 1e-4,
        'DROPOUT_RATE': 0.25,
        'MAX_EPOCHS': 500,
        'EARLY_STOPPING_PATIENCE': 50
    },
    # Higher learning rate
    {
        'name': 'higher_lr',
        'LEARNING_RATE': 0.02,
        'BATCH_SIZE': 128,
        'WEIGHT_DECAY': 1e-4,
        'DROPOUT_RATE': 0.25,
        'MAX_EPOCHS': 500,
        'EARLY_STOPPING_PATIENCE': 50
    },
    # Smaller batch size
    {
        'name': 'smaller_batch',
        'LEARNING_RATE': 0.01,
        'BATCH_SIZE': 64,
        'WEIGHT_DECAY': 1e-4,
        'DROPOUT_RATE': 0.25,
        'MAX_EPOCHS': 500,
        'EARLY_STOPPING_PATIENCE': 50
    },
    # Higher weight decay (stronger regularization)
    {
        'name': 'stronger_reg',
        'LEARNING_RATE': 0.01,
        'BATCH_SIZE': 128,
        'WEIGHT_DECAY': 5e-4,
        'DROPOUT_RATE': 0.3,
        'MAX_EPOCHS': 500,
        'EARLY_STOPPING_PATIENCE': 50
    },
    # Lower dropout
    {
        'name': 'lower_dropout',
        'LEARNING_RATE': 0.01,
        'BATCH_SIZE': 128,
        'WEIGHT_DECAY': 1e-4,
        'DROPOUT_RATE': 0.15,
        'MAX_EPOCHS': 500,
        'EARLY_STOPPING_PATIENCE': 50
    },
    # Longer training
    {
        'name': 'longer_training',
        'LEARNING_RATE': 0.01,
        'BATCH_SIZE': 128,
        'WEIGHT_DECAY': 1e-4,
        'DROPOUT_RATE': 0.25,
        'MAX_EPOCHS': 800,
        'EARLY_STOPPING_PATIENCE': 80
    },
    # Adaptive configuration 1
    {
        'name': 'adaptive_1',
        'LEARNING_RATE': 0.008,
        'BATCH_SIZE': 96,
        'WEIGHT_DECAY': 2e-4,
        'DROPOUT_RATE': 0.2,
        'MAX_EPOCHS': 600,
        'EARLY_STOPPING_PATIENCE': 60
    },
    # Adaptive configuration 2
    {
        'name': 'adaptive_2',
        'LEARNING_RATE': 0.015,
        'BATCH_SIZE': 96,
        'WEIGHT_DECAY': 1.5e-4,
        'DROPOUT_RATE': 0.22,
        'MAX_EPOCHS': 600,
        'EARLY_STOPPING_PATIENCE': 65
    }
]


def update_config_file(params):
    """Update config.py with new parameters"""
    config_path = Path('/home/vivian/eeg/EEG_experiments/config.py')
    with open(config_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        updated = False
        for key, value in params.items():
            if line.strip().startswith(f'{key} ='):
                new_lines.append(f'{key} = {value}\n')
                updated = True
                break
        if not updated:
            new_lines.append(line)

    with open(config_path, 'w') as f:
        f.writelines(new_lines)


def run_experiment(num_runs=1):
    """Run main_tfdwt.py and extract results"""
    results = []

    for run_idx in range(num_runs):
        print(f"\n{'='*60}")
        print(f"Run {run_idx + 1}/{num_runs}")
        print(f"{'='*60}")

        # Run the experiment
        result = subprocess.run(
            ['python', 'main_tfdwt.py'],
            capture_output=True,
            text=True,
            cwd='/home/vivian/eeg/EEG_experiments'
        )

        # Parse output for results
        output = result.stdout + result.stderr

        # Extract metrics from output
        p3_acc = None
        avo_acc = None
        overall_acc = None

        for line in output.split('\n'):
            if 'P3 Dataset - Mean Accuracy:' in line:
                try:
                    p3_acc = float(line.split('Mean Accuracy:')[1].split('|')[0].strip())
                except:
                    pass
            elif 'AVO Dataset - Mean Accuracy:' in line:
                try:
                    avo_acc = float(line.split('Mean Accuracy:')[1].split('|')[0].strip())
                except:
                    pass
            elif 'Overall accuracy:' in line and '±' in line:
                try:
                    overall_acc = float(line.split('Overall accuracy:')[1].split('±')[0].strip())
                except:
                    pass

        results.append({
            'run': run_idx + 1,
            'p3_accuracy': p3_acc,
            'avo_accuracy': avo_acc,
            'overall_accuracy': overall_acc,
            'timestamp': datetime.now().isoformat()
        })

        print(f"Run {run_idx + 1} Results:")
        print(f"  P3 Accuracy: {p3_acc}")
        print(f"  AVO Accuracy: {avo_acc}")
        print(f"  Overall Accuracy: {overall_acc}")

    return results


def check_stability(results, metric, threshold, required_runs=5):
    """Check if metric meets threshold for required consecutive runs"""
    if len(results) < required_runs:
        return False

    recent_results = results[-required_runs:]
    for r in recent_results:
        if r[metric] is None or r[metric] < threshold:
            return False
    return True


def save_results(all_results, filepath):
    """Save all results to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {filepath}")


def main():
    print("="*80)
    print("TF-DWT Optimization Script")
    print("="*80)

    results_dir = Path('/home/vivian/eeg/EEG_experiments/optimization_results')
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results = {}

    for scenario_name, scenario_config in SCENARIOS.items():
        print(f"\n{'#'*80}")
        print(f"# Scenario: {scenario_name}")
        print(f"# P3 trials: {scenario_config['NESTED_CV_TRIALS_PER_SUBJECT_P3']}")
        print(f"# AVO trials: {scenario_config['NESTED_CV_TRIALS_PER_SUBJECT_AVO']}")
        print(f"# Target: {scenario_config['target_metric']} >= {scenario_config['target_value']}")
        print(f"{'#'*80}")

        all_results[scenario_name] = {
            'config': scenario_config,
            'param_results': {}
        }

        # Update scenario-specific config
        update_config_file({
            'NESTED_CV_TRIALS_PER_SUBJECT_P3': scenario_config['NESTED_CV_TRIALS_PER_SUBJECT_P3'],
            'NESTED_CV_TRIALS_PER_SUBJECT_AVO': scenario_config['NESTED_CV_TRIALS_PER_SUBJECT_AVO']
        })

        target_met = False
        best_config = None
        best_avg_metric = 0

        for param_config in PARAM_CONFIGS:
            if target_met:
                break

            print(f"\n{'-'*80}")
            print(f"Testing configuration: {param_config['name']}")
            print(f"{'-'*80}")

            # Update parameters
            update_config_file({k: v for k, v in param_config.items() if k != 'name'})

            # Run experiments
            run_results = []
            max_runs = 10  # Maximum runs before moving to next config

            for run_idx in range(max_runs):
                print(f"\nConfiguration '{param_config['name']}' - Run {run_idx + 1}")

                single_run_results = run_experiment(num_runs=1)
                run_results.extend(single_run_results)

                # Check if we've achieved stability
                if len(run_results) >= 5:
                    is_stable = check_stability(
                        run_results,
                        scenario_config['target_metric'],
                        scenario_config['target_value'],
                        required_runs=5
                    )

                    if is_stable:
                        print(f"\n{'*'*80}")
                        print(f"SUCCESS! Configuration '{param_config['name']}' achieved target!")
                        print(f"Last 5 runs of {scenario_config['target_metric']}:")
                        for r in run_results[-5:]:
                            print(f"  Run {r['run']}: {r[scenario_config['target_metric']]:.4f}")
                        print(f"{'*'*80}")
                        target_met = True
                        best_config = param_config
                        break

                # Early stopping if performance is clearly insufficient
                if run_idx >= 2:
                    recent_avg = np.mean([r[scenario_config['target_metric']]
                                         for r in run_results[-3:]
                                         if r[scenario_config['target_metric']] is not None])
                    if recent_avg < scenario_config['target_value'] - 0.05:
                        print(f"Performance insufficient (avg: {recent_avg:.4f}), moving to next config...")
                        break

            # Track results
            avg_metric = np.mean([r[scenario_config['target_metric']]
                                 for r in run_results
                                 if r[scenario_config['target_metric']] is not None])

            all_results[scenario_name]['param_results'][param_config['name']] = {
                'params': param_config,
                'runs': run_results,
                'avg_metric': float(avg_metric) if not np.isnan(avg_metric) else None,
                'target_met': target_met
            }

            if avg_metric > best_avg_metric:
                best_avg_metric = avg_metric
                if not target_met:
                    best_config = param_config

            # Save intermediate results
            save_results(all_results, results_dir / f'optimization_{timestamp}.json')

        # Summary for this scenario
        print(f"\n{'='*80}")
        print(f"Scenario '{scenario_name}' Summary:")
        print(f"Target met: {target_met}")
        if best_config:
            print(f"Best configuration: {best_config['name']}")
            print(f"Best average {scenario_config['target_metric']}: {best_avg_metric:.4f}")
        print(f"{'='*80}")

    # Final summary
    print(f"\n{'#'*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'#'*80}")

    for scenario_name, scenario_data in all_results.items():
        print(f"\n{scenario_name}:")
        target_met = any(r['target_met'] for r in scenario_data['param_results'].values())
        print(f"  Target achieved: {target_met}")

        if target_met:
            successful_config = [name for name, r in scenario_data['param_results'].items()
                               if r['target_met']][0]
            print(f"  Successful configuration: {successful_config}")

    # Save final results
    save_results(all_results, results_dir / f'optimization_final_{timestamp}.json')

    # Generate summary report
    summary_path = results_dir / f'optimization_summary_{timestamp}.txt'
    with open(summary_path, 'w') as f:
        f.write("TF-DWT Optimization Summary\n")
        f.write("=" * 80 + "\n\n")

        for scenario_name, scenario_data in all_results.items():
            f.write(f"Scenario: {scenario_name}\n")
            f.write(f"Configuration: P3={scenario_data['config']['NESTED_CV_TRIALS_PER_SUBJECT_P3']}, ")
            f.write(f"AVO={scenario_data['config']['NESTED_CV_TRIALS_PER_SUBJECT_AVO']}\n")
            f.write(f"Target: {scenario_data['config']['target_metric']} >= {scenario_data['config']['target_value']}\n\n")

            for config_name, config_data in scenario_data['param_results'].items():
                f.write(f"  Configuration: {config_name}\n")
                f.write(f"    Parameters: {config_data['params']}\n")
                f.write(f"    Average metric: {config_data['avg_metric']}\n")
                f.write(f"    Target met: {config_data['target_met']}\n")
                f.write(f"    Number of runs: {len(config_data['runs'])}\n\n")

            f.write("\n" + "-" * 80 + "\n\n")

    print(f"\nSummary report saved to {summary_path}")
    print("\nOptimization complete!")


if __name__ == '__main__':
    main()
