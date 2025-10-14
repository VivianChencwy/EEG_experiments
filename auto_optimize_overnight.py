#!/usr/bin/env python3
"""
Automated overnight optimization script
Runs multiple experiments with different configurations
"""

import subprocess
import time
import json
import re
from datetime import datetime
from pathlib import Path

def extract_results_from_log(log_path):
    """Extract P3 and AVO accuracy from log file"""
    if not Path(log_path).exists():
        return None, None, None

    with open(log_path, 'r') as f:
        content = f.read()

    p3_acc = None
    avo_acc = None
    overall_acc = None

    # Extract from final summary
    for line in content.split('\n'):
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

    return p3_acc, avo_acc, overall_acc


def update_config(param_dict):
    """Update config.py with new parameters"""
    config_path = Path('/home/vivian/eeg/EEG_experiments/config.py')

    with open(config_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        updated = False
        for key, value in param_dict.items():
            if line.strip().startswith(f'{key} ='):
                if isinstance(value, str) and not value.replace('.','').replace('-','').isdigit():
                    new_lines.append(f'{key} = "{value}"\n')
                else:
                    new_lines.append(f'{key} = {value}\n')
                updated = True
                break
        if not updated:
            new_lines.append(line)

    with open(config_path, 'w') as f:
        f.writelines(new_lines)


def run_experiment(config_name):
    """Run a single experiment and return results"""
    print(f"\n{'='*70}")
    print(f"Running experiment: {config_name}")
    print(f"{'='*70}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'exp_{config_name}_{timestamp}.log'

    start_time = time.time()

    # Run experiment
    result = subprocess.run(
        ['conda', 'run', '-n', 'eeg', 'python', 'main_tfdwt.py'],
        capture_output=True,
        text=True,
        cwd='/home/vivian/eeg/EEG_experiments'
    )

    elapsed = time.time() - start_time

    # Save log
    with open(log_file, 'w') as f:
        f.write(result.stdout)
        f.write(result.stderr)

    # Extract results
    p3_acc, avo_acc, overall_acc = extract_results_from_log(log_file)

    print(f"Completed in {elapsed/60:.1f} minutes")
    print(f"  P3: {p3_acc:.4f if p3_acc else 'N/A'}")
    print(f"  AVO: {avo_acc:.4f if avo_acc else 'N/A'}")
    print(f"  Overall: {overall_acc:.4f if overall_acc else 'N/A'}")

    return {
        'config_name': config_name,
        'timestamp': timestamp,
        'log_file': log_file,
        'p3_accuracy': p3_acc,
        'avo_accuracy': avo_acc,
        'overall_accuracy': overall_acc,
        'elapsed_minutes': elapsed/60
    }


def main():
    print("="*70)
    print("Overnight Automated Optimization")
    print("="*70)

    results_log = []

    # Scenario 1: AVO-focused (P3=80, AVO=10, target AVO>=0.66)
    print("\n" + "#"*70)
    print("# SCENARIO 1: AVO-Focused (P3=80, AVO=10)")
    print("#"*70)

    avo_configs = [
        {
            'name': 'avo_opt1',
            'params': {
                'NESTED_CV_TRIALS_PER_SUBJECT_P3': 80,
                'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 10,
                'LEARNING_RATE': 0.006,
                'BATCH_SIZE': 80,
                'WEIGHT_DECAY': 2.5e-4,
                'DROPOUT_RATE': 0.18,
                'MAX_EPOCHS': 700,
                'EARLY_STOPPING_PATIENCE': 80
            }
        },
        {
            'name': 'avo_opt2',
            'params': {
                'NESTED_CV_TRIALS_PER_SUBJECT_P3': 80,
                'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 10,
                'LEARNING_RATE': 0.012,
                'BATCH_SIZE': 64,
                'WEIGHT_DECAY': 1.5e-4,
                'DROPOUT_RATE': 0.15,
                'MAX_EPOCHS': 650,
                'EARLY_STOPPING_PATIENCE': 75
            }
        },
        {
            'name': 'avo_opt3',
            'params': {
                'NESTED_CV_TRIALS_PER_SUBJECT_P3': 80,
                'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 10,
                'LEARNING_RATE': 0.005,
                'BATCH_SIZE': 112,
                'WEIGHT_DECAY': 3e-4,
                'DROPOUT_RATE': 0.17,
                'MAX_EPOCHS': 750,
                'EARLY_STOPPING_PATIENCE': 85
            }
        },
    ]

    avo_success = False
    avo_results = []

    for config in avo_configs:
        update_config(config['params'])

        # Run 5 times to check stability
        run_results = []
        for run_idx in range(5):
            print(f"\nConfig '{config['name']}' - Run {run_idx + 1}/5")
            result = run_experiment(f"{config['name']}_run{run_idx+1}")
            run_results.append(result)
            results_log.append(result)

            # Check if we have 5 successful runs
            if len(run_results) >= 5:
                avo_accs = [r['avo_accuracy'] for r in run_results[-5:] if r['avo_accuracy'] is not None]
                if len(avo_accs) == 5 and all(acc >= 0.66 for acc in avo_accs):
                    print(f"\n{'*'*70}")
                    print(f"SUCCESS! Config '{config['name']}' achieved AVO target!")
                    print(f"Last 5 AVO accuracies: {[f'{a:.4f}' for a in avo_accs]}")
                    print(f"{'*'*70}")
                    avo_success = True
                    avo_results = run_results
                    break

        if avo_success:
            break

    # Scenario 2: P3-focused (P3=10, AVO=80, target P3>=0.62)
    print("\n" + "#"*70)
    print("# SCENARIO 2: P3-Focused (P3=10, AVO=80)")
    print("#"*70)

    p3_configs = [
        {
            'name': 'p3_opt1',
            'params': {
                'NESTED_CV_TRIALS_PER_SUBJECT_P3': 10,
                'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 80,
                'LEARNING_RATE': 0.007,
                'BATCH_SIZE': 80,
                'WEIGHT_DECAY': 2e-4,
                'DROPOUT_RATE': 0.19,
                'MAX_EPOCHS': 650,
                'EARLY_STOPPING_PATIENCE': 75
            }
        },
        {
            'name': 'p3_opt2',
            'params': {
                'NESTED_CV_TRIALS_PER_SUBJECT_P3': 10,
                'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 80,
                'LEARNING_RATE': 0.01,
                'BATCH_SIZE': 64,
                'WEIGHT_DECAY': 1.8e-4,
                'DROPOUT_RATE': 0.16,
                'MAX_EPOCHS': 700,
                'EARLY_STOPPING_PATIENCE': 80
            }
        },
        {
            'name': 'p3_opt3',
            'params': {
                'NESTED_CV_TRIALS_PER_SUBJECT_P3': 10,
                'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 80,
                'LEARNING_RATE': 0.005,
                'BATCH_SIZE': 96,
                'WEIGHT_DECAY': 2.2e-4,
                'DROPOUT_RATE': 0.18,
                'MAX_EPOCHS': 750,
                'EARLY_STOPPING_PATIENCE': 85
            }
        },
    ]

    p3_success = False
    p3_results = []

    for config in p3_configs:
        update_config(config['params'])

        # Run 5 times to check stability
        run_results = []
        for run_idx in range(5):
            print(f"\nConfig '{config['name']}' - Run {run_idx + 1}/5")
            result = run_experiment(f"{config['name']}_run{run_idx+1}")
            run_results.append(result)
            results_log.append(result)

            # Check if we have 5 successful runs
            if len(run_results) >= 5:
                p3_accs = [r['p3_accuracy'] for r in run_results[-5:] if r['p3_accuracy'] is not None]
                if len(p3_accs) == 5 and all(acc >= 0.62 for acc in p3_accs):
                    print(f"\n{'*'*70}")
                    print(f"SUCCESS! Config '{config['name']}' achieved P3 target!")
                    print(f"Last 5 P3 accuracies: {[f'{a:.4f}' for a in p3_accs]}")
                    print(f"{'*'*70}")
                    p3_success = True
                    p3_results = run_results
                    break

        if p3_success:
            break

    # Save all results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'optimization_results_{timestamp}.json'

    with open(results_file, 'w') as f:
        json.dump({
            'avo_scenario': {
                'success': avo_success,
                'results': avo_results
            },
            'p3_scenario': {
                'success': p3_success,
                'results': p3_results
            },
            'all_results': results_log
        }, f, indent=2)

    # Final summary
    print("\n" + "="*70)
    print("OVERNIGHT OPTIMIZATION SUMMARY")
    print("="*70)
    print(f"\nTotal experiments run: {len(results_log)}")
    print(f"\nAVO Scenario (P3=80, AVO=10, target AVO>=0.66): {avo_success}")
    print(f"P3 Scenario (P3=10, AVO=80, target P3>=0.62): {p3_success}")
    print(f"\nResults saved to: {results_file}")

    # Create summary report
    summary_file = f'optimization_summary_{timestamp}.txt'
    with open(summary_file, 'w') as f:
        f.write("Overnight Optimization Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total experiments: {len(results_log)}\n\n")

        f.write("AVO Scenario Results:\n")
        f.write(f"  Success: {avo_success}\n")
        if avo_results:
            f.write("  Last 5 runs:\n")
            for r in avo_results[-5:]:
                f.write(f"    - AVO: {r['avo_accuracy']:.4f}, P3: {r['p3_accuracy']:.4f}\n")
        f.write("\n")

        f.write("P3 Scenario Results:\n")
        f.write(f"  Success: {p3_success}\n")
        if p3_results:
            f.write("  Last 5 runs:\n")
            for r in p3_results[-5:]:
                f.write(f"    - P3: {r['p3_accuracy']:.4f}, AVO: {r['avo_accuracy']:.4f}\n")

    print(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()
