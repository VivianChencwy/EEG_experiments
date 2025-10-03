#!/usr/bin/env python
"""
Quick optimization script - tests baseline then applies targeted improvements
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import re

def update_config(params):
    """Update config.py parameters"""
    config_path = '/home/vivian/eeg/EEG_experiments/config.py'

    with open(config_path, 'r') as f:
        content = f.read()

    for key, value in params.items():
        # Match the parameter line and replace it
        pattern = rf'^{key}\s*=\s*.*$'
        if isinstance(value, str):
            replacement = f'{key} = "{value}"'
        else:
            replacement = f'{key} = {value}'
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    with open(config_path, 'w') as f:
        f.write(content)

    print(f"Updated config: {params}")


def run_single_experiment():
    """Run one experiment and extract key metrics"""
    print("\nRunning experiment...")
    start_time = time.time()

    result = subprocess.run(
        ['conda', 'run', '-n', 'eeg', 'python', 'main_tfdwt.py'],
        capture_output=True,
        text=True,
        cwd='/home/vivian/eeg/EEG_experiments'
    )

    elapsed = time.time() - start_time
    output = result.stdout + result.stderr

    # Extract metrics
    p3_acc = None
    avo_acc = None
    overall_acc = None

    for line in output.split('\n'):
        if 'P3 Dataset - Mean Accuracy:' in line:
            match = re.search(r'Mean Accuracy:\s*([\d.]+)', line)
            if match:
                p3_acc = float(match.group(1))
        elif 'AVO Dataset - Mean Accuracy:' in line:
            match = re.search(r'Mean Accuracy:\s*([\d.]+)', line)
            if match:
                avo_acc = float(match.group(1))
        elif 'Overall accuracy:' in line and '±' in line:
            match = re.search(r'Overall accuracy:\s*([\d.]+)', line)
            if match:
                overall_acc = float(match.group(1))

    print(f"Completed in {elapsed/60:.1f} minutes")
    print(f"  P3: {p3_acc:.4f if p3_acc else 'N/A'}")
    print(f"  AVO: {avo_acc:.4f if avo_acc else 'N/A'}")
    print(f"  Overall: {overall_acc:.4f if overall_acc else 'N/A'}")

    return {'p3': p3_acc, 'avo': avo_acc, 'overall': overall_acc, 'time': elapsed}


def test_configuration(params, target_metric, target_value, max_runs=5, required_stable_runs=5):
    """Test a configuration until stable or max runs reached"""
    print(f"\n{'='*70}")
    print(f"Testing configuration: {params.get('name', 'unnamed')}")
    print(f"{'='*70}")

    # Update config
    config_params = {k: v for k, v in params.items() if k != 'name'}
    update_config(config_params)

    results = []

    for run in range(max_runs):
        print(f"\n--- Run {run + 1}/{max_runs} ---")
        result = run_single_experiment()
        results.append(result)

        # Check stability after we have enough runs
        if len(results) >= required_stable_runs:
            recent = [r[target_metric] for r in results[-required_stable_runs:]]
            if all(v is not None and v >= target_value for v in recent):
                print(f"\n✓ SUCCESS! Stable performance achieved:")
                print(f"  Last {required_stable_runs} runs: {[f'{v:.4f}' for v in recent]}")
                return True, results

    # Check if we're close
    avg = np.mean([r[target_metric] for r in results if r[target_metric] is not None])
    print(f"\nAverage {target_metric}: {avg:.4f} (target: {target_value:.4f})")

    return False, results


def main():
    print("="*70)
    print("TF-DWT Quick Optimization")
    print("="*70)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'/home/vivian/eeg/EEG_experiments/optimization_log_{timestamp}.txt'

    # Redirect output to both console and file
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    log_f = open(log_file, 'w')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_f)

    try:
        # Scenario 1: AVO-focused (P3=80, AVO=10 -> AVO acc >= 0.66)
        print("\n" + "#"*70)
        print("# SCENARIO 1: AVO-Focused (P3=80, AVO=10)")
        print("# Target: AVO accuracy >= 0.66 (5 stable runs)")
        print("#"*70)

        # Progressive configurations for AVO
        avo_configs = [
            {
                'name': 'baseline_avo',
                'NESTED_CV_TRIALS_PER_SUBJECT_P3': 80,
                'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 10,
                'LEARNING_RATE': 0.01,
                'BATCH_SIZE': 128,
                'WEIGHT_DECAY': 1e-4,
                'DROPOUT_RATE': 0.25,
                'MAX_EPOCHS': 500,
                'EARLY_STOPPING_PATIENCE': 50
            },
            {
                'name': 'avo_tuned_1',
                'NESTED_CV_TRIALS_PER_SUBJECT_P3': 80,
                'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 10,
                'LEARNING_RATE': 0.008,
                'BATCH_SIZE': 96,
                'WEIGHT_DECAY': 2e-4,
                'DROPOUT_RATE': 0.2,
                'MAX_EPOCHS': 600,
                'EARLY_STOPPING_PATIENCE': 60
            },
            {
                'name': 'avo_tuned_2',
                'NESTED_CV_TRIALS_PER_SUBJECT_P3': 80,
                'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 10,
                'LEARNING_RATE': 0.005,
                'BATCH_SIZE': 64,
                'WEIGHT_DECAY': 3e-4,
                'DROPOUT_RATE': 0.18,
                'MAX_EPOCHS': 700,
                'EARLY_STOPPING_PATIENCE': 70
            },
            {
                'name': 'avo_tuned_3',
                'NESTED_CV_TRIALS_PER_SUBJECT_P3': 80,
                'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 10,
                'LEARNING_RATE': 0.012,
                'BATCH_SIZE': 80,
                'WEIGHT_DECAY': 1.5e-4,
                'DROPOUT_RATE': 0.22,
                'MAX_EPOCHS': 650,
                'EARLY_STOPPING_PATIENCE': 65
            }
        ]

        avo_success = False
        avo_best_config = None

        for config in avo_configs:
            success, results = test_configuration(
                config,
                target_metric='avo',
                target_value=0.66,
                max_runs=8,
                required_stable_runs=5
            )
            if success:
                avo_success = True
                avo_best_config = config
                break

        # Scenario 2: P3-focused (P3=10, AVO=80 -> P3 acc >= 0.62)
        print("\n" + "#"*70)
        print("# SCENARIO 2: P3-Focused (P3=10, AVO=80)")
        print("# Target: P3 accuracy >= 0.62 (5 stable runs)")
        print("#"*70)

        p3_configs = [
            {
                'name': 'baseline_p3',
                'NESTED_CV_TRIALS_PER_SUBJECT_P3': 10,
                'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 80,
                'LEARNING_RATE': 0.01,
                'BATCH_SIZE': 128,
                'WEIGHT_DECAY': 1e-4,
                'DROPOUT_RATE': 0.25,
                'MAX_EPOCHS': 500,
                'EARLY_STOPPING_PATIENCE': 50
            },
            {
                'name': 'p3_tuned_1',
                'NESTED_CV_TRIALS_PER_SUBJECT_P3': 10,
                'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 80,
                'LEARNING_RATE': 0.008,
                'BATCH_SIZE': 96,
                'WEIGHT_DECAY': 2e-4,
                'DROPOUT_RATE': 0.2,
                'MAX_EPOCHS': 600,
                'EARLY_STOPPING_PATIENCE': 60
            },
            {
                'name': 'p3_tuned_2',
                'NESTED_CV_TRIALS_PER_SUBJECT_P3': 10,
                'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 80,
                'LEARNING_RATE': 0.006,
                'BATCH_SIZE': 64,
                'WEIGHT_DECAY': 2.5e-4,
                'DROPOUT_RATE': 0.19,
                'MAX_EPOCHS': 650,
                'EARLY_STOPPING_PATIENCE': 65
            },
            {
                'name': 'p3_tuned_3',
                'NESTED_CV_TRIALS_PER_SUBJECT_P3': 10,
                'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 80,
                'LEARNING_RATE': 0.01,
                'BATCH_SIZE': 80,
                'WEIGHT_DECAY': 1.8e-4,
                'DROPOUT_RATE': 0.21,
                'MAX_EPOCHS': 600,
                'EARLY_STOPPING_PATIENCE': 60
            }
        ]

        p3_success = False
        p3_best_config = None

        for config in p3_configs:
            success, results = test_configuration(
                config,
                target_metric='p3',
                target_value=0.62,
                max_runs=8,
                required_stable_runs=5
            )
            if success:
                p3_success = True
                p3_best_config = config
                break

        # Final Summary
        print("\n" + "="*70)
        print("OPTIMIZATION SUMMARY")
        print("="*70)

        print("\nScenario 1 (AVO-focused):")
        print(f"  Success: {avo_success}")
        if avo_best_config:
            print(f"  Best config: {avo_best_config['name']}")
            print(f"  Parameters: {avo_best_config}")

        print("\nScenario 2 (P3-focused):")
        print(f"  Success: {p3_success}")
        if p3_best_config:
            print(f"  Best config: {p3_best_config['name']}")
            print(f"  Parameters: {p3_best_config}")

        # Save successful configs
        if avo_success or p3_success:
            summary_file = f'/home/vivian/eeg/EEG_experiments/successful_configs_{timestamp}.txt'
            with open(summary_file, 'w') as f:
                f.write("Successful TF-DWT Configurations\n")
                f.write("="*70 + "\n\n")

                if avo_success:
                    f.write("AVO-Focused Configuration (P3=80, AVO=10 -> AVO>=0.66):\n")
                    for k, v in avo_best_config.items():
                        f.write(f"  {k} = {v}\n")
                    f.write("\n")

                if p3_success:
                    f.write("P3-Focused Configuration (P3=10, AVO=80 -> P3>=0.62):\n")
                    for k, v in p3_best_config.items():
                        f.write(f"  {k} = {v}\n")

            print(f"\nSuccessful configurations saved to: {summary_file}")

    finally:
        sys.stdout = original_stdout
        log_f.close()
        print(f"\nLog saved to: {log_file}")


if __name__ == '__main__':
    main()
