#!/usr/bin/env python3
"""
Quick TF-DWT Parameter Test

This script runs a minimal test to verify the parameter optimization works.
Uses very reduced CV settings for rapid feedback.
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent

# Minimal test parameters - just test key ones
TEST_PARAMS = [
    {
        'w_small_cap': 3.0,  # Current default
        'mmd_thresholds': (2.0, 4.0, 0.1, 0.2, 0.3),  # Current default
        'guard_factors': (0.8, 0.5, 0.7),  # Current default
        'warmup_config': (2, 5, 0.1),  # Current default
        'early_stop_patience': 30,  # Reduced
        'learning_rate': 0.01,
    },
    {
        'w_small_cap': 2.5,  # Lower cap - potentially better for small domain
        'mmd_thresholds': (1.5, 3.0, 0.05, 0.15, 0.25),  # Lower thresholds/weights
        'guard_factors': (0.8, 0.5, 0.7),
        'warmup_config': (2, 5, 0.1),
        'early_stop_patience': 30,
        'learning_rate': 0.012,  # Slightly higher LR
    },
]

def read_base_config() -> str:
    return (PROJECT_ROOT / 'config.py').read_text(encoding='utf-8')

def apply_config_overrides(config_text: str, overrides: dict) -> str:
    """Apply configuration overrides to config text"""
    import re
    lines = config_text.splitlines()

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

    for key, value in overrides.items():
        if key in ['use_combined_datasets']:
            set_line(key, 'True' if value else 'False')
        elif key in ['data_dir']:
            set_line(key, str(value))
        elif key in ['dataset']:
            set_line(key, repr(value))
        elif key.startswith('NESTED_CV') or key in ['EARLY_STOPPING_PATIENCE', 'LEARNING_RATE']:
            set_line(key, str(value))

    return "\n".join(lines) + "\n"

def create_modified_tfdwt(params: dict) -> str:
    """Create a modified version of main_tfdwt.py with test parameters"""
    tfdwt_path = PROJECT_ROOT / 'main_tfdwt.py'
    original_content = tfdwt_path.read_text(encoding='utf-8')
    modified_content = original_content

    # Apply modifications
    w_cap = params['w_small_cap']
    modified_content = modified_content.replace(
        'w_small_target = min(w_small_target, 3.0)',
        f'w_small_target = min(w_small_target, {w_cap})'
    )

    thresh1, thresh2, lambda1, lambda2, lambda3 = params['mmd_thresholds']
    old_lambda = 'lambda_mmd = 0.1 if overall_ratio < 2.0 else (0.2 if overall_ratio < 4.0 else 0.3)'
    new_lambda = f'lambda_mmd = {lambda1} if overall_ratio < {thresh1} else ({lambda2} if overall_ratio < {thresh2} else {lambda3})'
    modified_content = modified_content.replace(old_lambda, new_lambda)

    return modified_content

def run_quick_test(params: dict, test_id: str, focus_case: str):
    """Run a single quick test"""
    print(f"\n{'='*50}")
    print(f"QUICK TEST {test_id}: {focus_case}")
    print(f"{'='*50}")

    # Test configuration
    if focus_case == 'P3_small':
        config_overrides = {
            'use_combined_datasets': True,
            'data_dir': 'P3_DATA_DIR',
            'dataset': 'use_combined_datasets',
            'NESTED_CV_TRIALS_PER_SUBJECT_P3': 20,
            'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 200,
            'NESTED_CV_OUTER_FOLDS': 2,  # Minimal CV
            'NESTED_CV_REPEATS': 1,      # Single run
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
            'EARLY_STOPPING_PATIENCE': params['early_stop_patience'],
            'LEARNING_RATE': params['learning_rate'],
        }

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"tfdwt_test_{test_id}_"))

    try:
        # Create modified config
        base_config = read_base_config()
        modified_config = apply_config_overrides(base_config, config_overrides)
        (tmp_dir / 'config.py').write_text(modified_config, encoding='utf-8')

        # Create modified main_tfdwt.py
        modified_tfdwt = create_modified_tfdwt(params)
        (tmp_dir / 'main_tfdwt.py').write_text(modified_tfdwt, encoding='utf-8')

        # Run test
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{str(tmp_dir)}:{str(PROJECT_ROOT)}"
        env['CONFIG_OVERRIDE_PATH'] = str(tmp_dir / 'config.py')

        python_code = f"""
import sys
import os
sys.path.insert(0, '{str(tmp_dir)}')
os.environ['CONFIG_OVERRIDE_PATH'] = '{str(tmp_dir / "config.py")}'
import runpy
runpy.run_path('{str(tmp_dir / "main_tfdwt.py")}', run_name='__main__')
"""

        cmd = [sys.executable, '-c', python_code]

        print(f"Starting test with parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")

        # Run with timeout
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
                # Print key progress
                line = output.strip()
                if any(keyword in line.lower() for keyword in [
                    'fold', 'accuracy', 'completed', 'final results'
                ]):
                    print(f"[{test_id}] {line}")

        # Extract result
        accuracy = 0.0
        for line in output_lines:
            if 'Final Results: Overall Accuracy' in line:
                import re
                match = re.search(r'Overall Accuracy = ([\d.]+)', line)
                if match:
                    accuracy = float(match.group(1))
                    break

        print(f"\n✅ Test {test_id} completed: Accuracy = {accuracy:.4f}")
        return {
            'test_id': test_id,
            'focus_case': focus_case,
            'accuracy': accuracy,
            'params': params,
            'return_code': proc.returncode
        }

    except Exception as e:
        print(f"❌ Test {test_id} failed: {e}")
        return {
            'test_id': test_id,
            'focus_case': focus_case,
            'accuracy': 0.0,
            'params': params,
            'error': str(e),
            'return_code': -1
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def main():
    print("🧪 Quick TF-DWT Parameter Test")
    print(f"Testing {len(TEST_PARAMS)} parameter combinations")

    results = []

    # Test both focus cases with both parameter sets
    for i, params in enumerate(TEST_PARAMS, 1):
        for focus_case in ['P3_small', 'AVO_small']:
            test_id = f"{focus_case}_T{i}"
            result = run_quick_test(params, test_id, focus_case)
            results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("QUICK TEST SUMMARY")
    print(f"{'='*60}")

    for result in results:
        status = "✅" if result['accuracy'] > 0 else "❌"
        print(f"{status} {result['test_id']}: {result['accuracy']:.4f}")

    # Find best results
    p3_results = [r for r in results if r['focus_case'] == 'P3_small']
    avo_results = [r for r in results if r['focus_case'] == 'AVO_small']

    if p3_results:
        best_p3 = max(p3_results, key=lambda x: x['accuracy'])
        print(f"\n🏆 Best P3_small: {best_p3['accuracy']:.4f} (Test {best_p3['test_id']})")

    if avo_results:
        best_avo = max(avo_results, key=lambda x: x['accuracy'])
        print(f"🏆 Best AVO_small: {best_avo['accuracy']:.4f} (Test {best_avo['test_id']})")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = PROJECT_ROOT / f"quick_test_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📋 Results saved to: {results_file}")
    return results

if __name__ == '__main__':
    main()