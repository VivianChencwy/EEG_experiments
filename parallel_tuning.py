#!/usr/bin/env python3
"""
Parallel Hyperparameter Tuning for TF-DWT

This script runs multiple tuning processes in parallel to speed up the search.
Each process explores a different region of the parameter space.

Usage:
    python parallel_tuning.py --n_processes 4 --trials_per_process 25

This will run 4 parallel processes, each doing 25 trials, for a total of 100 trials.
"""

import argparse
import subprocess
import sys
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List
import numpy as np


def run_tuning_subprocess(process_id: int, n_trials: int, base_seed: int) -> Dict[str, Any]:
    """Run tuning in a subprocess with specific seed for reproducibility."""

    # Create process-specific results directory
    results_dir = f"parallel_tuning_results/process_{process_id}"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Set different random seed for each process
    np.random.seed(base_seed + process_id * 1000)

    print(f"Starting process {process_id} with {n_trials} trials...")

    # Run the tuning
    cmd = [
        sys.executable, "tune_tfdwt.py",
        "--strategy", "random",
        "--n_trials", str(n_trials),
        "--results_dir", results_dir
    ]

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)  # 4 hour timeout
        duration = time.time() - start_time

        if result.returncode == 0:
            # Load results
            results_file = Path(results_dir) / "tuning_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    tuning_results = json.load(f)

                return {
                    'process_id': process_id,
                    'success': True,
                    'duration': duration,
                    'best_score': tuning_results.get('best_score', 0.0),
                    'best_params': tuning_results.get('best_params', {}),
                    'n_trials': tuning_results.get('n_trials', 0),
                    'results_file': str(results_file)
                }
            else:
                return {
                    'process_id': process_id,
                    'success': False,
                    'duration': duration,
                    'error': 'Results file not found'
                }
        else:
            return {
                'process_id': process_id,
                'success': False,
                'duration': duration,
                'error': result.stderr
            }

    except subprocess.TimeoutExpired:
        return {
            'process_id': process_id,
            'success': False,
            'duration': time.time() - start_time,
            'error': 'Process timeout'
        }
    except Exception as e:
        return {
            'process_id': process_id,
            'success': False,
            'duration': time.time() - start_time,
            'error': str(e)
        }


def combine_results(process_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine results from all parallel processes."""

    # Collect all successful results
    all_trials = []
    best_overall_score = -float('inf')
    best_overall_params = None
    total_trials = 0
    successful_processes = 0

    for proc_result in process_results:
        if proc_result['success']:
            successful_processes += 1
            total_trials += proc_result['n_trials']

            # Load detailed results
            try:
                with open(proc_result['results_file'], 'r') as f:
                    detailed_results = json.load(f)

                # Add trials from this process
                for trial in detailed_results.get('trials', []):
                    trial['source_process'] = proc_result['process_id']
                    all_trials.append(trial)

                # Update global best
                if proc_result['best_score'] > best_overall_score:
                    best_overall_score = proc_result['best_score']
                    best_overall_params = proc_result['best_params']

            except Exception as e:
                print(f"Warning: Could not load detailed results from process {proc_result['process_id']}: {e}")

    return {
        'total_processes': len(process_results),
        'successful_processes': successful_processes,
        'total_trials': total_trials,
        'best_score': best_overall_score,
        'best_params': best_overall_params,
        'all_trials': all_trials,
        'process_results': process_results
    }


def save_combined_results(combined_results: Dict[str, Any], output_dir: str):
    """Save combined results to files."""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save combined results JSON
    results_file = output_path / "parallel_tuning_combined_results.json"
    with open(results_file, 'w') as f:
        json.dump(combined_results, f, indent=2)

    # Save best config as Python file
    if combined_results['best_params']:
        from tune_tfdwt import TFDWTTuner
        tuner = TFDWTTuner()
        best_config_path = tuner.create_config_for_trial(combined_results['best_params'], 'parallel_best')
        import shutil
        final_best_path = output_path / "best_config_parallel.py"
        shutil.copy(best_config_path, final_best_path)
        Path(best_config_path).unlink()  # Clean up temp file

    # Generate summary report
    report = generate_parallel_report(combined_results)
    report_file = output_path / "parallel_tuning_report.md"
    with open(report_file, 'w') as f:
        f.write(report)

    return str(results_file), str(report_file)


def generate_parallel_report(combined_results: Dict[str, Any]) -> str:
    """Generate a summary report of parallel tuning results."""

    report = f"""
# Parallel Hyperparameter Tuning Report for TF-DWT

## Summary
- Total processes: {combined_results['total_processes']}
- Successful processes: {combined_results['successful_processes']}
- Total trials across all processes: {combined_results['total_trials']}
- Best overall accuracy: {combined_results['best_score']:.4f}

## Process Performance
"""

    # Process performance table
    for proc_result in combined_results['process_results']:
        status = "✓ Success" if proc_result['success'] else "✗ Failed"
        duration_str = f"{proc_result['duration']/60:.1f} min"

        if proc_result['success']:
            score_str = f"{proc_result['best_score']:.4f}"
            trials_str = f"{proc_result['n_trials']} trials"
        else:
            score_str = "N/A"
            trials_str = f"Error: {proc_result.get('error', 'Unknown')}"

        report += f"- Process {proc_result['process_id']}: {status} | {duration_str} | Score: {score_str} | {trials_str}\n"

    # Best parameters
    if combined_results['best_params']:
        report += f"""
## Best Parameters (Accuracy: {combined_results['best_score']:.4f})
```json
{json.dumps(combined_results['best_params'], indent=2)}
```
"""

    # Performance distribution analysis
    if combined_results['all_trials']:
        scores = [t['score'] for t in combined_results['all_trials'] if t['score'] > 0]
        if scores:
            report += f"""
## Performance Statistics
- Mean accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}
- Median accuracy: {np.median(scores):.4f}
- 95th percentile: {np.percentile(scores, 95):.4f}
- Number of successful trials: {len(scores)}

## Top 10 Results Across All Processes
"""
            # Sort all trials by score
            sorted_trials = sorted(combined_results['all_trials'],
                                  key=lambda x: x['score'], reverse=True)

            for i, trial in enumerate(sorted_trials[:10]):
                if trial['score'] > 0:
                    report += f"\n### Rank {i+1}: {trial['score']:.4f} (Process {trial['source_process']})\n"
                    # Show only key parameters for brevity
                    key_params = {k: v for k, v in trial['params'].items()
                                 if k in ['LEARNING_RATE', 'classifier', 'w_small_clip_max', 'lambda_mmd_base']}
                    report += f"```json\n{json.dumps(key_params, indent=2)}\n```\n"

    return report


def main():
    parser = argparse.ArgumentParser(description='Parallel Hyperparameter Tuning for TF-DWT')
    parser.add_argument('--n_processes', type=int, default=4,
                        help='Number of parallel processes to run')
    parser.add_argument('--trials_per_process', type=int, default=25,
                        help='Number of trials per process')
    parser.add_argument('--output_dir', default='parallel_tuning_results',
                        help='Output directory for combined results')
    parser.add_argument('--base_seed', type=int, default=42,
                        help='Base random seed (each process gets base_seed + process_id * 1000)')

    args = parser.parse_args()

    # Verify required files
    if not Path("tune_tfdwt.py").exists():
        print("ERROR: tune_tfdwt.py not found")
        return 1

    if not Path("main_tfdwt.py").exists():
        print("ERROR: main_tfdwt.py not found")
        return 1

    print(f"Starting parallel tuning with {args.n_processes} processes")
    print(f"Each process will run {args.trials_per_process} trials")
    print(f"Total trials: {args.n_processes * args.trials_per_process}")
    print(f"Estimated time: {args.n_processes * args.trials_per_process * 5 / 60:.1f} - {args.n_processes * args.trials_per_process * 10 / 60:.1f} hours")

    start_time = time.time()

    # Run parallel processes
    with ProcessPoolExecutor(max_workers=args.n_processes) as executor:
        # Submit all processes
        future_to_process = {
            executor.submit(run_tuning_subprocess, i, args.trials_per_process, args.base_seed): i
            for i in range(args.n_processes)
        }

        # Collect results as they complete
        process_results = []
        completed = 0

        for future in as_completed(future_to_process):
            process_id = future_to_process[future]
            try:
                result = future.result()
                process_results.append(result)
                completed += 1

                status = "SUCCESS" if result['success'] else "FAILED"
                elapsed = time.time() - start_time
                print(f"Process {process_id} completed ({completed}/{args.n_processes}) - {status} - "
                      f"Elapsed: {elapsed/60:.1f}min")

                if result['success']:
                    print(f"  Best score: {result['best_score']:.4f}")
                else:
                    print(f"  Error: {result.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"Process {process_id} failed with exception: {e}")
                process_results.append({
                    'process_id': process_id,
                    'success': False,
                    'error': str(e),
                    'duration': time.time() - start_time
                })

    # Combine and save results
    print("\nCombining results from all processes...")
    combined_results = combine_results(process_results)

    results_file, report_file = save_combined_results(combined_results, args.output_dir)

    total_time = time.time() - start_time

    # Print summary
    print(f"\n{'='*60}")
    print("PARALLEL HYPERPARAMETER TUNING COMPLETED")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Successful processes: {combined_results['successful_processes']}/{combined_results['total_processes']}")
    print(f"Total successful trials: {len([t for t in combined_results['all_trials'] if t['score'] > 0])}")
    print(f"Best accuracy found: {combined_results['best_score']:.4f}")
    print(f"Results saved to: {results_file}")
    print(f"Report saved to: {report_file}")

    if combined_results['best_params']:
        print(f"Best config saved to: {args.output_dir}/best_config_parallel.py")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)