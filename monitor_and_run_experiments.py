#!/usr/bin/env python3
"""
Monitor current experiment and run subsequent ones automatically
"""

import subprocess
import time
import sys
from pathlib import Path
from datetime import datetime
import csv

def get_latest_results():
    """Get results from the most recent TF-DWT CSV"""
    csv_files = sorted(Path('.').glob('tfdwt_summary_stats_*.csv'), key=lambda x: x.stat().st_mtime, reverse=True)

    if not csv_files:
        return None, None, None

    latest_csv = csv_files[0]

    with open(latest_csv, 'r') as f:
        reader = csv.DictReader(f)
        row = list(reader)[0]

        p3_acc = float(row['mean_p3_accuracy'])
        avo_acc = float(row['mean_avo_accuracy'])
        overall_acc = float(row['mean_accuracy'])

    return p3_acc, avo_acc, overall_acc


def wait_for_experiment():
    """Wait for current experiment to complete"""
    print("Waiting for current experiment to complete...")

    while True:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )

        if 'python main_tfdwt.py' not in result.stdout:
            print("Experiment completed!")
            break

        time.sleep(30)  # Check every 30 seconds

    # Wait a bit for files to be written
    time.sleep(5)


def run_experiment(run_number):
    """Run a single experiment"""
    print(f"\n{'='*70}")
    print(f"Starting Experiment {run_number}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # Run in foreground to ensure completion
    result = subprocess.run(
        ['conda', 'run', '-n', 'eeg', 'python', 'main_tfdwt.py'],
        capture_output=False,  # Let output go to terminal
        cwd='/home/vivian/eeg/EEG_experiments'
    )

    # Get results
    p3_acc, avo_acc, overall_acc = get_latest_results()

    print(f"\n{'='*70}")
    print(f"Experiment {run_number} Results:")
    print(f"  P3: {p3_acc:.4f if p3_acc else 'N/A'}")
    print(f"  AVO: {avo_acc:.4f if avo_acc else 'N/A'}")
    print(f"  Overall: {overall_acc:.4f if overall_acc else 'N/A'}")
    print(f"{'='*70}\n")

    return p3_acc, avo_acc, overall_acc


def main():
    print("="*70)
    print("Automated Experiment Runner")
    print("="*70)

    # Wait for current experiment if running
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    if 'python main_tfdwt.py' in result.stdout:
        wait_for_experiment()

        # Get results from completed experiment
        p3_acc, avo_acc, overall_acc = get_latest_results()
        print(f"\nCompleted experiment results:")
        print(f"  P3: {p3_acc:.4f if p3_acc is not None else 'N/A'}")
        print(f"  AVO: {avo_acc:.4f if avo_acc is not None else 'N/A'}")
        print(f"  Overall: {overall_acc:.4f if overall_acc is not None else 'N/A'}")

        results = [(p3_acc, avo_acc, overall_acc)]
        start_run = 2
    else:
        results = []
        start_run = 1

    # Run remaining experiments
    for run_num in range(start_run, 6):
        p3, avo, overall = run_experiment(run_num)
        results.append((p3, avo, overall))

        # Check if we have 5 successful AVO results
        if len(results) >= 5:
            avo_accs = [r[1] for r in results[-5:] if r[1] is not None]
            if len(avo_accs) == 5 and all(acc >= 0.66 for acc in avo_accs):
                print(f"\n{'*'*70}")
                print("SUCCESS! 5 consecutive runs achieved AVO >= 0.66!")
                print(f"AVO accuracies: {[f'{a:.4f}' for a in avo_accs]}")
                print(f"{'*'*70}\n")
                break

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal runs completed: {len(results)}")

    for i, (p3, avo, overall) in enumerate(results, 1):
        print(f"Run {i}: P3={p3:.4f if p3 else 'N/A'}, AVO={avo:.4f if avo else 'N/A'}, Overall={overall:.4f if overall else 'N/A'}")

    # Check AVO target
    if len(results) >= 5:
        avo_accs = [r[1] for r in results[-5:] if r[1] is not None]
        if len(avo_accs) == 5:
            all_pass = all(acc >= 0.66 for acc in avo_accs)
            print(f"\nAVO Target (>= 0.66): {'✓ MET' if all_pass else '✗ NOT MET'}")

    # Save summary
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = f'experiment_summary_{timestamp}.txt'

    with open(summary_file, 'w') as f:
        f.write("Automated Experiment Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total runs: {len(results)}\n\n")

        for i, (p3, avo, overall) in enumerate(results, 1):
            f.write(f"Run {i}:\n")
            f.write(f"  P3: {p3:.4f if p3 else 'N/A'}\n")
            f.write(f"  AVO: {avo:.4f if avo else 'N/A'}\n")
            f.write(f"  Overall: {overall:.4f if overall else 'N/A'}\n\n")

    print(f"\nSummary saved to: {summary_file}")


if __name__ == '__main__':
    main()
