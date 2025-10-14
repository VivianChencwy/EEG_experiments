#!/usr/bin/env python3
"""
Run TF-DWT experiments with optimal parameters found from minimal testing.

Based on minimal test results:
- P3_small: conservative parameters work best
- AVO_small: aggressive parameters work best
"""

import os
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent

# Optimal parameters from minimal testing
OPTIMAL_PARAMS = {
    'P3_small': {
        'name': 'conservative',
        'params': {
            'w_small_cap': 2.5,
            'mmd_thresholds': (1.5, 3.0, 0.05, 0.15, 0.25),
            'guard_factors': (0.8, 0.5, 0.7),
            'warmup_config': (2, 5, 0.1),
            'early_stop_patience': 15,
            'learning_rate': 0.012,
        }
    },
    'AVO_small': {
        'name': 'aggressive',
        'params': {
            'w_small_cap': 4.0,
            'mmd_thresholds': (2.5, 5.0, 0.15, 0.25, 0.35),
            'guard_factors': (0.7, 0.4, 0.6),
            'warmup_config': (3, 7, 0.12),
            'early_stop_patience': 25,
            'learning_rate': 0.008,
        }
    }
}

def create_config_with_params(focus_case: str, params: dict) -> str:
    """Create config.py content with optimal parameters"""
    base_config = (PROJECT_ROOT / 'config.py').read_text(encoding='utf-8')

    # Configuration based on focus case
    if focus_case == 'P3_small':
        config_overrides = {
            'use_combined_datasets': True,
            'data_dir': 'P3_DATA_DIR',
            'dataset': 'use_combined_datasets',
            'NESTED_CV_TRIALS_PER_SUBJECT_P3': 20,   # Small P3
            'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 200, # Large AVO
            'NESTED_CV_OUTER_FOLDS': 5,              # Full CV
            'NESTED_CV_REPEATS': 5,
            'EARLY_STOPPING_PATIENCE': params['early_stop_patience'],
            'LEARNING_RATE': params['learning_rate'],
        }
    else:  # AVO_small
        config_overrides = {
            'use_combined_datasets': True,
            'data_dir': 'P3_DATA_DIR',
            'dataset': 'use_combined_datasets',
            'NESTED_CV_TRIALS_PER_SUBJECT_P3': 200,  # Large P3
            'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 20,  # Small AVO
            'NESTED_CV_OUTER_FOLDS': 5,
            'NESTED_CV_REPEATS': 5,
            'EARLY_STOPPING_PATIENCE': params['early_stop_patience'],
            'LEARNING_RATE': params['learning_rate'],
        }

    # Apply overrides
    import re
    lines = base_config.splitlines()

    def set_line(prefix: str, value_src: str):
        nonlocal lines
        pat = re.compile(rf"^({re.escape(prefix)}\s*=).*$")
        replaced = False
        for i, line in enumerate(lines):
            if pat.match(line.strip()):
                lines[i] = f"{prefix} = {value_src}"
                replaced = True
                break
        if not replaced:
            lines.append(f"{prefix} = {value_src}")

    for key, value in config_overrides.items():
        if key in ['use_combined_datasets']:
            set_line(key, 'True' if value else 'False')
        elif key in ['data_dir']:
            set_line(key, str(value))
        elif key in ['dataset']:
            set_line(key, repr(value))
        else:
            set_line(key, str(value))

    return "\n".join(lines) + "\n"

def create_optimized_tfdwt(params: dict) -> str:
    """Create modified main_tfdwt.py with optimal parameters"""
    tfdwt_path = PROJECT_ROOT / 'main_tfdwt.py'
    original_content = tfdwt_path.read_text(encoding='utf-8')
    modified_content = original_content

    # Apply optimal parameter modifications
    w_cap = params['w_small_cap']
    modified_content = modified_content.replace(
        'w_small_target = min(w_small_target, 3.0)',
        f'w_small_target = min(w_small_target, {w_cap})'
    )

    thresh1, thresh2, lambda1, lambda2, lambda3 = params['mmd_thresholds']
    old_lambda = 'lambda_mmd = 0.1 if overall_ratio < 2.0 else (0.2 if overall_ratio < 4.0 else 0.3)'
    new_lambda = f'lambda_mmd = {lambda1} if overall_ratio < {thresh1} else ({lambda2} if overall_ratio < {thresh2} else {lambda3})'
    modified_content = modified_content.replace(old_lambda, new_lambda)

    small_w_decay, small_mmd_decay, large_mmd_decay = params['guard_factors']
    modified_content = modified_content.replace(
        'new_w = max(1.0, cur_w * 0.8)',
        f'new_w = max(1.0, cur_w * {small_w_decay})'
    )
    modified_content = modified_content.replace(
        'new_lambda = max(0.0, cur_lambda * 0.5)',
        f'new_lambda = max(0.0, cur_lambda * {small_mmd_decay})'
    )
    modified_content = modified_content.replace(
        'new_lambda = max(0.0, cur_lambda * 0.7)',
        f'new_lambda = max(0.0, cur_lambda * {large_mmd_decay})'
    )

    min_epochs, max_epochs, ratio = params['warmup_config']
    old_warmup = 'warmup = max(2, min(5, int(0.1 * MAX_EPOCHS)))'
    new_warmup = f'warmup = max({min_epochs}, min({max_epochs}, int({ratio} * MAX_EPOCHS)))'
    modified_content = modified_content.replace(old_warmup, new_warmup)

    return modified_content

def run_optimal_experiment(focus_case: str) -> dict:
    """Run experiment with optimal parameters"""
    optimal_config = OPTIMAL_PARAMS[focus_case]
    params = optimal_config['params']
    param_name = optimal_config['name']

    print(f"\n{'='*60}")
    print(f"OPTIMAL EXPERIMENT: {focus_case}")
    print(f"Parameter set: {param_name}")
    print(f"{'='*60}")

    # Create temporary directory
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"optimal_{focus_case}_{param_name}_"))

    try:
        # Create optimized config
        config_content = create_config_with_params(focus_case, params)
        (tmp_dir / 'config.py').write_text(config_content, encoding='utf-8')

        # Create optimized main_tfdwt.py
        tfdwt_content = create_optimized_tfdwt(params)
        (tmp_dir / 'main_tfdwt.py').write_text(tfdwt_content, encoding='utf-8')

        # Setup environment
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{str(tmp_dir)}:{str(PROJECT_ROOT)}"
        env['CONFIG_OVERRIDE_PATH'] = str(tmp_dir / 'config.py')

        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"optimal_{focus_case}_{param_name}_{timestamp}.log"
        log_path = PROJECT_ROOT / log_filename

        python_code = f"""
import sys
import os
sys.path.insert(0, '{str(tmp_dir)}')
os.environ['CONFIG_OVERRIDE_PATH'] = '{str(tmp_dir / "config.py")}'
import runpy
runpy.run_path('{str(tmp_dir / "main_tfdwt.py")}', run_name='__main__')
"""

        cmd = [sys.executable, '-c', python_code]

        print("Starting optimized TF-DWT experiment...")
        print("Optimal parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")

        print(f"\nLog file: {log_path}")

        # Run experiment
        with open(log_path, 'w', encoding='utf-8') as log_file:
            proc = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1
            )

            output_lines = []
            while True:
                output = proc.stdout.readline()
                if output == '' and proc.poll() is not None:
                    break
                if output:
                    log_file.write(output)
                    log_file.flush()
                    output_lines.append(output.strip())

                    # Print progress
                    line = output.strip()
                    if any(keyword in line.lower() for keyword in [
                        'fold', 'repeat', 'accuracy', 'completed', 'final results', 'saved to'
                    ]):
                        print(f"[{focus_case}] {line}")

            proc.wait()

        # Extract results
        final_accuracy = 0.0
        csv_files = []

        for line in output_lines:
            if 'Final Results: Overall Accuracy' in line:
                import re
                match = re.search(r'Overall Accuracy = ([\d.]+)', line)
                if match:
                    final_accuracy = float(match.group(1))

            # Look for CSV files
            if 'detailed results saved to:' in line.lower() or 'tfdwt_detailed_results_' in line:
                import re
                csv_match = re.search(r'tfdwt_detailed_results_\d+_\d+\.csv', line)
                if csv_match:
                    csv_file = PROJECT_ROOT / csv_match.group(0)
                    if csv_file.exists():
                        csv_files.append(csv_file)

        result = {
            'focus_case': focus_case,
            'parameter_set': param_name,
            'parameters': params,
            'overall_accuracy': final_accuracy,
            'log_file': str(log_path),
            'csv_files': [str(f) for f in csv_files],
            'return_code': proc.returncode,
        }

        print(f"\n✅ {focus_case} COMPLETED:")
        print(f"  Overall accuracy: {final_accuracy:.4f}")
        print(f"  Log: {log_path.name}")
        if csv_files:
            print(f"  CSV: {csv_files[0].name}")

        return result

    except Exception as e:
        print(f"❌ {focus_case} FAILED: {e}")
        return {
            'focus_case': focus_case,
            'overall_accuracy': 0.0,
            'error': str(e),
            'return_code': -1
        }
    finally:
        # Clean up
        shutil.rmtree(tmp_dir, ignore_errors=True)

def analyze_optimization_results(results: list):
    """Analyze and compare optimization results with baseline"""
    import pandas as pd

    print(f"\n{'='*80}")
    print("OPTIMIZATION RESULTS ANALYSIS")
    print(f"{'='*80}")

    # Baseline accuracies (small datasets)
    baseline_accuracies = {
        'P3_small': 0.5853,  # P3 accuracy when P3=20, AVO=200
        'AVO_small': 0.6749  # AVO accuracy when P3=200, AVO=20
    }

    for result in results:
        if result['return_code'] != 0 or not result.get('csv_files'):
            continue

        focus_case = result['focus_case']
        param_set = result['parameter_set']

        # Load detailed CSV results
        csv_file = result['csv_files'][0]
        df = pd.read_csv(csv_file)

        if focus_case == 'P3_small':
            # P3 is the small dataset
            small_dataset_acc = df['p3_accuracy'].mean()
            small_dataset_std = df['p3_accuracy'].std()
            large_dataset_acc = df['avo_accuracy'].mean()
        else:
            # AVO is the small dataset
            small_dataset_acc = df['avo_accuracy'].mean()
            small_dataset_std = df['avo_accuracy'].std()
            large_dataset_acc = df['p3_accuracy'].mean()

        baseline_acc = baseline_accuracies[focus_case]
        improvement = small_dataset_acc - baseline_acc
        improvement_pct = (improvement / baseline_acc) * 100

        print(f"\n🎯 {focus_case} ({param_set} parameters):")
        print(f"  Small dataset accuracy: {small_dataset_acc:.4f} ± {small_dataset_std:.4f}")
        print(f"  Baseline accuracy: {baseline_acc:.4f}")
        print(f"  Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")
        print(f"  Large dataset accuracy: {large_dataset_acc:.4f}")
        print(f"  Overall accuracy: {result['overall_accuracy']:.4f}")

        # Statistical significance check
        import scipy.stats as stats
        if len(df) > 1:
            t_stat, p_value = stats.ttest_1samp(df[f"{'p3' if focus_case == 'P3_small' else 'avo'}_accuracy"], baseline_acc)
            significance = "✅ Significant" if p_value < 0.05 else "⚠️  Not significant"
            print(f"  Statistical test: {significance} (p={p_value:.4f})")

def main():
    """Run optimal TF-DWT experiments"""
    print("🚀 Running TF-DWT Experiments with Optimal Parameters")
    print("Based on minimal test results:")
    print("  P3_small: conservative parameters")
    print("  AVO_small: aggressive parameters")

    results = []

    # Run both experiments
    for focus_case in ['P3_small', 'AVO_small']:
        result = run_optimal_experiment(focus_case)
        results.append(result)

    # Analyze results
    analyze_optimization_results(results)

    # Save results summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    import json
    results_file = PROJECT_ROOT / f"optimal_tfdwt_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📁 Results summary saved to: {results_file}")

    return results

if __name__ == '__main__':
    main()