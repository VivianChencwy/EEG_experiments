#!/usr/bin/env python3
"""
Ultimate P3 optimization strategy for extreme imbalance (P3=10, AVO=80)
Target: P3 accuracy > 0.6

This implements the most aggressive possible parameter combinations
to maximize P3 performance in severely imbalanced scenarios.
"""

import sys
import os
import subprocess
import time
from datetime import datetime
import itertools

def run_tfdwt_experiment(params, experiment_name):
    """Run TF-DWT experiment with given parameters"""
    print(f"\n=== Running {experiment_name} ===")
    print(f"Parameters: {params}")

    # Create parameter string for main_tfdwt.py
    param_str = (
        f"--w_small_cap {params['w_small_cap']} "
        f"--mmd_alpha {params['mmd_thresholds'][0]} "
        f"--mmd_beta {params['mmd_thresholds'][1]} "
        f"--mmd_gamma {params['mmd_thresholds'][2]} "
        f"--mmd_delta {params['mmd_thresholds'][3]} "
        f"--mmd_epsilon {params['mmd_thresholds'][4]} "
        f"--guard_factor_1 {params['guard_factors'][0]} "
        f"--guard_factor_2 {params['guard_factors'][1]} "
        f"--warmup_epochs {params['warmup_config']['warmup_epochs']} "
        f"--warmup_lr_scale {params['warmup_config']['warmup_lr_scale']} "
        f"--warmup_weight_scale {params['warmup_config']['warmup_weight_scale']} "
        f"--learning_rate {params['learning_rate']} "
        f"--batch_size {params['batch_size']}"
    )

    # Run the experiment
    cmd = f"python3 main_tfdwt.py combined P3_10 AVO_80 {param_str}"
    print(f"Command: {cmd}")

    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.1f} seconds")

    if result.returncode != 0:
        print(f"Error running experiment: {result.stderr}")
        return None

    print("Experiment completed successfully")
    return result.stdout

def get_latest_result_file():
    """Get the most recent TF-DWT result file"""
    import glob
    files = glob.glob("tfdwt_detailed_results_*.csv")
    if not files:
        return None
    return max(files, key=os.path.getctime)

def extract_p3_accuracy(result_file):
    """Extract P3 accuracy from result file"""
    if not result_file or not os.path.exists(result_file):
        return None

    try:
        import pandas as pd
        df = pd.read_csv(result_file)
        p3_accuracy = df['p3_accuracy'].mean()
        return p3_accuracy
    except Exception as e:
        print(f"Error reading result file: {e}")
        return None

def main():
    """Main optimization pipeline"""
    print("=== Ultimate P3 Optimization for P3=10, AVO=80 ===")
    print("Target: P3 accuracy > 0.6")

    # Ultimate parameter combinations - even more extreme than before
    ultimate_params = [
        {
            'name': 'EXTREME_BOOST',
            'w_small_cap': 12.0,  # Much higher boost
            'mmd_thresholds': (0.3, 0.6, 0.005, 0.02, 0.05),  # Very tight alignment
            'guard_factors': (0.02, 0.05),  # Minimal guards
            'warmup_config': {
                'warmup_epochs': 30,
                'warmup_lr_scale': 0.2,
                'warmup_weight_scale': 0.3
            },
            'learning_rate': 0.025,
            'batch_size': 8  # Smaller batch for more updates
        },
        {
            'name': 'MAXIMUM_WEIGHT',
            'w_small_cap': 15.0,  # Maximum reasonable boost
            'mmd_thresholds': (0.2, 0.4, 0.003, 0.015, 0.03),  # Even tighter
            'guard_factors': (0.01, 0.03),  # Almost no guards
            'warmup_config': {
                'warmup_epochs': 35,
                'warmup_lr_scale': 0.15,
                'warmup_weight_scale': 0.25
            },
            'learning_rate': 0.03,
            'batch_size': 4  # Very small batch
        },
        {
            'name': 'PRECISION_FOCUS',
            'w_small_cap': 10.0,
            'mmd_thresholds': (0.1, 0.2, 0.001, 0.005, 0.01),  # Ultra-tight alignment
            'guard_factors': (0.005, 0.01),  # Extremely minimal guards
            'warmup_config': {
                'warmup_epochs': 40,
                'warmup_lr_scale': 0.1,
                'warmup_weight_scale': 0.2
            },
            'learning_rate': 0.035,
            'batch_size': 2  # Minimal batch size
        }
    ]

    best_params = None
    best_accuracy = 0.0
    results = []

    for params in ultimate_params:
        print(f"\n{'='*60}")
        print(f"Testing configuration: {params['name']}")

        # Record baseline
        baseline_file = get_latest_result_file()

        # Run experiment
        output = run_tfdwt_experiment(params, params['name'])

        if output is None:
            print(f"Failed to run {params['name']}")
            continue

        # Get results
        result_file = get_latest_result_file()
        p3_accuracy = extract_p3_accuracy(result_file)

        if p3_accuracy is None:
            print(f"Failed to extract P3 accuracy for {params['name']}")
            continue

        print(f"P3 accuracy: {p3_accuracy:.4f}")
        print(f"Target achieved: {p3_accuracy > 0.6}")

        results.append({
            'name': params['name'],
            'params': params,
            'p3_accuracy': p3_accuracy,
            'target_achieved': p3_accuracy > 0.6,
            'result_file': result_file
        })

        if p3_accuracy > best_accuracy:
            best_accuracy = p3_accuracy
            best_params = params

        # If we achieved target, we can stop
        if p3_accuracy > 0.6:
            print(f"\\n🎉 TARGET ACHIEVED! P3 accuracy: {p3_accuracy:.4f} > 0.6")
            break

    # Summary
    print(f"\\n{'='*60}")
    print("ULTIMATE OPTIMIZATION SUMMARY")
    print(f"{'='*60}")

    for result in results:
        status = "✅ TARGET ACHIEVED" if result['target_achieved'] else "❌ Below target"
        print(f"{result['name']}: {result['p3_accuracy']:.4f} - {status}")

    if best_params:
        print(f"\\nBest configuration: {best_params['name']}")
        print(f"Best P3 accuracy: {best_accuracy:.4f}")
        print(f"Target achieved: {best_accuracy > 0.6}")

        # Save best parameters
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"ultimate_p3_params_{timestamp}.py", "w") as f:
            f.write(f"# Ultimate P3 optimization results - {timestamp}\\n")
            f.write(f"# Best P3 accuracy: {best_accuracy:.4f}\\n")
            f.write(f"# Target achieved: {best_accuracy > 0.6}\\n\\n")
            f.write(f"ULTIMATE_P3_PARAMS = {best_params}\\n")

        print(f"\\nBest parameters saved to: ultimate_p3_params_{timestamp}.py")

    return best_accuracy > 0.6

if __name__ == "__main__":
    success = main()
    if success:
        print("\\n🎉 Mission accomplished! P3 accuracy > 0.6 achieved!")
    else:
        print("\\n⚠️  Target not yet achieved. May need even more extreme parameters.")

    sys.exit(0 if success else 1)