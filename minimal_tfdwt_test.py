#!/usr/bin/env python3
"""
Minimal TF-DWT Parameter Test

This script runs the absolute minimum test to verify our parameter
modification approach works and shows meaningful differences.

Uses extremely reduced CV: 1 fold, 1 repeat for rapid feedback.
"""

import os
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent

# Just test 3 key parameter combinations
TEST_CASES = [
    {
        'name': 'baseline',
        'params': {
            'w_small_cap': 3.0,
            'mmd_thresholds': (2.0, 4.0, 0.1, 0.2, 0.3),
            'guard_factors': (0.8, 0.5, 0.7),
            'warmup_config': (2, 5, 0.1),
            'early_stop_patience': 20,
            'learning_rate': 0.01,
        }
    },
    {
        'name': 'conservative',
        'params': {
            'w_small_cap': 2.5,  # Lower cap
            'mmd_thresholds': (1.5, 3.0, 0.05, 0.15, 0.25),  # More conservative MMD
            'guard_factors': (0.8, 0.5, 0.7),
            'warmup_config': (2, 5, 0.1),
            'early_stop_patience': 15,  # Earlier stopping
            'learning_rate': 0.012,  # Slightly higher LR
        }
    },
    {
        'name': 'aggressive',
        'params': {
            'w_small_cap': 4.0,  # Higher cap
            'mmd_thresholds': (2.5, 5.0, 0.15, 0.25, 0.35),  # More aggressive MMD
            'guard_factors': (0.7, 0.4, 0.6),  # More aggressive guards
            'warmup_config': (3, 7, 0.12),  # Longer warmup
            'early_stop_patience': 25,
            'learning_rate': 0.008,  # Lower LR to be more careful
        }
    }
]


def create_minimal_config(focus_case: str, params: dict) -> str:
    """Create minimal config for rapid testing"""
    base_config = (PROJECT_ROOT / 'config.py').read_text(encoding='utf-8')

    # Configuration based on focus case
    if focus_case == 'P3_small':
        config_overrides = {
            'use_combined_datasets': True,
            'data_dir': 'P3_DATA_DIR',
            'dataset': 'use_combined_datasets',
            'NESTED_CV_TRIALS_PER_SUBJECT_P3': 20,
            'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 200,
            'NESTED_CV_OUTER_FOLDS': 2,  # Minimal CV
            'NESTED_CV_REPEATS': 1,      # Single run
            'MAX_EPOCHS': 50,            # Faster training
            'EARLY_STOPPING_PATIENCE': params['early_stop_patience'],
            'LEARNING_RATE': params['learning_rate'],
        }
    else:
        config_overrides = {
            'use_combined_datasets': True,
            'data_dir': 'P3_DATA_DIR',
            'dataset': 'use_combined_datasets',
            'NESTED_CV_TRIALS_PER_SUBJECT_P3': 200,
            'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 20,
            'NESTED_CV_OUTER_FOLDS': 2,
            'NESTED_CV_REPEATS': 1,
            'MAX_EPOCHS': 50,
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


def create_minimal_tfdwt(params: dict) -> str:
    """Create modified main_tfdwt.py for testing"""
    tfdwt_path = PROJECT_ROOT / 'main_tfdwt.py'
    original_content = tfdwt_path.read_text(encoding='utf-8')
    modified_content = original_content

    # Apply parameter modifications
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


def run_minimal_test(focus_case: str, test_case: dict) -> dict:
    """Run a single minimal test"""
    case_name = test_case['name']
    params = test_case['params']

    print(f"\n{'='*40}")
    print(f"TEST: {case_name} ({focus_case})")
    print(f"{'='*40}")

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"minimal_{case_name}_"))

    try:
        # Create modified files
        config_content = create_minimal_config(focus_case, params)
        (tmp_dir / 'config.py').write_text(config_content, encoding='utf-8')

        tfdwt_content = create_minimal_tfdwt(params)
        (tmp_dir / 'main_tfdwt.py').write_text(tfdwt_content, encoding='utf-8')

        # Setup environment
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{str(tmp_dir)}:{str(PROJECT_ROOT)}"
        env['CONFIG_OVERRIDE_PATH'] = str(tmp_dir / 'config.py')

        # Run experiment
        python_code = f"""
import sys
import os
sys.path.insert(0, '{str(tmp_dir)}')
os.environ['CONFIG_OVERRIDE_PATH'] = '{str(tmp_dir / "config.py")}'
import runpy
runpy.run_path('{str(tmp_dir / "main_tfdwt.py")}', run_name='__main__')
"""

        cmd = [sys.executable, '-c', python_code]

        print(f"Parameters: w_cap={params['w_small_cap']}, lr={params['learning_rate']}")
        print("Running minimal experiment...")

        start_time = datetime.now()

        proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        output_lines = []
        while True:
            output = proc.stdout.readline()
            if output == '' and proc.poll() is not None:
                break
            if output:
                output_lines.append(output.strip())
                # Print key lines
                line = output.strip()
                if any(keyword in line.lower() for keyword in [
                    'fold', 'accuracy', 'final results', 'completed', 'processing'
                ]):
                    print(f"  {line}")

        end_time = datetime.now()
        duration = end_time - start_time

        # Extract accuracy
        accuracy = 0.0
        for line in output_lines:
            if 'Final Results: Overall Accuracy' in line:
                import re
                match = re.search(r'Overall Accuracy = ([\d.]+)', line)
                if match:
                    accuracy = float(match.group(1))
                    break

        result = {
            'case_name': case_name,
            'focus_case': focus_case,
            'accuracy': accuracy,
            'duration_seconds': duration.total_seconds(),
            'parameters': params.copy(),
            'return_code': proc.returncode,
        }

        status = "✅" if accuracy > 0 else "❌"
        print(f"{status} Result: {accuracy:.4f} (took {duration.total_seconds():.1f}s)")

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            'case_name': case_name,
            'focus_case': focus_case,
            'accuracy': 0.0,
            'error': str(e),
            'return_code': -1
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    """Run minimal tests for both focus cases"""
    print("🧪 Minimal TF-DWT Parameter Tests")
    print("Testing 3 parameter combinations on both focus cases")

    all_results = []

    for focus_case in ['P3_small', 'AVO_small']:
        print(f"\n{'='*60}")
        print(f"TESTING: {focus_case}")
        print(f"{'='*60}")

        for test_case in TEST_CASES:
            result = run_minimal_test(focus_case, test_case)
            all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("MINIMAL TEST RESULTS SUMMARY")
    print(f"{'='*60}")

    # Results by focus case
    for focus_case in ['P3_small', 'AVO_small']:
        case_results = [r for r in all_results if r['focus_case'] == focus_case]
        valid_results = [r for r in case_results if r['accuracy'] > 0]

        print(f"\n📊 {focus_case}:")
        if valid_results:
            # Sort by accuracy
            sorted_results = sorted(valid_results, key=lambda x: x['accuracy'], reverse=True)

            best = sorted_results[0]
            worst = sorted_results[-1] if len(sorted_results) > 1 else best

            print(f"  🏆 Best: {best['case_name']} = {best['accuracy']:.4f}")
            if len(sorted_results) > 1:
                print(f"  📉 Worst: {worst['case_name']} = {worst['accuracy']:.4f}")
                print(f"  📈 Range: {best['accuracy'] - worst['accuracy']:.4f}")

            print("  All results:")
            for result in sorted_results:
                print(f"    {result['case_name']}: {result['accuracy']:.4f}")
        else:
            print(f"  ❌ No successful results")

    # Parameter analysis
    print(f"\n📈 PARAMETER EFFECTIVENESS:")

    # Group by parameter combination
    param_groups = {}
    for result in all_results:
        if result['accuracy'] > 0:
            case_name = result['case_name']
            if case_name not in param_groups:
                param_groups[case_name] = []
            param_groups[case_name].append(result['accuracy'])

    for case_name, accuracies in param_groups.items():
        mean_acc = sum(accuracies) / len(accuracies)
        print(f"  {case_name}: {mean_acc:.4f} (avg across both cases)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = PROJECT_ROOT / f"minimal_test_results_{timestamp}.json"

    import json
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n📁 Results saved to: {results_file.name}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")

    if param_groups:
        # Find best overall parameter combination
        best_overall = max(param_groups.items(), key=lambda x: sum(x[1])/len(x[1]))
        print(f"  • Best overall parameter set: {best_overall[0]}")

        # Find parameter set with least variance
        variances = {}
        for case_name, accuracies in param_groups.items():
            if len(accuracies) > 1:
                import numpy as np
                variances[case_name] = np.var(accuracies)

        if variances:
            most_stable = min(variances.items(), key=lambda x: x[1])
            print(f"  • Most stable parameter set: {most_stable[0]} (variance: {most_stable[1]:.6f})")

        print(f"  • Next step: Run full optimization with promising parameter ranges")
    else:
        print(f"  • Check implementation - no successful results")

    return all_results


if __name__ == '__main__':
    main()