#!/usr/bin/env python3
"""
TF-DWT Optimization Results Analysis

This script analyzes the results from staged optimization and creates
final configurations and experiment scripts for the best parameters.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / 'log_staged_optimization'


def find_latest_results(focus_case: str) -> Optional[Path]:
    """Find the most recent optimization results for a focus case"""
    pattern = f"staged_optimization_{focus_case}_*.json"
    json_files = list(RESULTS_DIR.glob(pattern))

    if not json_files:
        return None

    # Return the most recent file
    return max(json_files, key=lambda x: x.stat().st_mtime)


def load_optimization_results(focus_case: str) -> Optional[Dict]:
    """Load optimization results from JSON file"""
    results_file = find_latest_results(focus_case)

    if not results_file:
        print(f"❌ No results found for {focus_case}")
        return None

    print(f"📁 Loading results from: {results_file.name}")

    with open(results_file, 'r') as f:
        data = json.load(f)

    return data


def analyze_stage_results(results_by_stage: Dict, focus_case: str) -> Dict:
    """Analyze results across all stages"""
    analysis = {
        'focus_case': focus_case,
        'stages': {},
        'best_overall': None,
        'parameter_trends': {},
    }

    all_results = []

    for stage, stage_results in results_by_stage.items():
        valid_results = [r for r in stage_results if r.get('accuracy', 0) > 0]

        if valid_results:
            best_result = max(valid_results, key=lambda x: x['accuracy'])
            accuracies = [r['accuracy'] for r in valid_results]

            stage_analysis = {
                'total_experiments': len(stage_results),
                'successful_experiments': len(valid_results),
                'success_rate': len(valid_results) / len(stage_results),
                'best_accuracy': best_result['accuracy'],
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'best_parameters': best_result['parameters'],
                'cv_config': best_result.get('cv_config', {}),
            }

            analysis['stages'][stage] = stage_analysis
            all_results.extend(valid_results)

    # Overall best
    if all_results:
        overall_best = max(all_results, key=lambda x: x['accuracy'])
        analysis['best_overall'] = overall_best

    # Parameter trends analysis
    analysis['parameter_trends'] = analyze_parameter_trends(all_results)

    return analysis


def analyze_parameter_trends(results: List[Dict]) -> Dict:
    """Analyze which parameters tend to work better"""
    trends = {}

    if not results:
        return trends

    # Group results by parameter values
    param_groups = {}

    for result in results:
        params = result.get('parameters', {})
        accuracy = result.get('accuracy', 0)

        for param_name, param_value in params.items():
            if param_name not in param_groups:
                param_groups[param_name] = {}

            # Convert tuples to strings for grouping
            key = str(param_value) if isinstance(param_value, (list, tuple)) else param_value

            if key not in param_groups[param_name]:
                param_groups[param_name] = {}

            if key not in param_groups[param_name]:
                param_groups[param_name][key] = []

            param_groups[param_name][key].append(accuracy)

    # Calculate statistics for each parameter value
    for param_name, param_values in param_groups.items():
        trends[param_name] = {}

        for value, accuracies in param_values.items():
            trends[param_name][value] = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'count': len(accuracies),
                'max_accuracy': max(accuracies),
            }

    return trends


def create_final_experiment_script(p3_best: Dict, avo_best: Dict) -> Path:
    """Create final experiment script with optimal parameters"""

    script_content = f'''#!/usr/bin/env python3
"""
Final TF-DWT Experiments with Optimized Parameters
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Optimized parameters found through staged optimization:

P3_small case - Best accuracy: {p3_best['accuracy']:.4f}
Parameters: {p3_best['parameters']}

AVO_small case - Best accuracy: {avo_best['accuracy']:.4f}
Parameters: {avo_best['parameters']}
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

# Optimal parameters from optimization
OPTIMAL_PARAMS = {{
    'P3_small': {json.dumps(p3_best, indent=8)},
    'AVO_small': {json.dumps(avo_best, indent=8)},
}}

FINAL_LOG_DIR = PROJECT_ROOT / 'log_final_optimized_experiments'
FINAL_LOG_DIR.mkdir(exist_ok=True)


def create_modified_config(base_config: str, config_overrides: dict) -> str:
    """Apply configuration overrides"""
    import re
    lines = base_config.splitlines()

    def set_line(prefix: str, value_src: str):
        nonlocal lines
        pat = re.compile(rf"^({{re.escape(prefix)}}\\s*=).*$")
        replaced = False
        for i, line in enumerate(lines):
            if pat.match(line.strip()):
                lines[i] = f"{{prefix}} = {{value_src}}"
                replaced = True
                break
        if not replaced:
            lines.append(f"{{prefix}} = {{value_src}}")

    for key, value in config_overrides.items():
        if key in ['use_combined_datasets']:
            set_line(key, 'True' if value else 'False')
        elif key in ['data_dir']:
            set_line(key, str(value))
        elif key in ['dataset']:
            set_line(key, repr(value))
        else:
            set_line(key, str(value))

    return "\\n".join(lines) + "\\n"


def create_modified_tfdwt(params: dict) -> str:
    """Create modified main_tfdwt.py with optimal parameters"""
    tfdwt_path = PROJECT_ROOT / 'main_tfdwt.py'
    original_content = tfdwt_path.read_text(encoding='utf-8')
    modified_content = original_content

    # Apply optimal parameter modifications
    w_cap = params['w_small_cap']
    modified_content = modified_content.replace(
        'w_small_target = min(w_small_target, 3.0)',
        f'w_small_target = min(w_small_target, {{w_cap}})'
    )

    thresh1, thresh2, lambda1, lambda2, lambda3 = params['mmd_thresholds']
    old_lambda = 'lambda_mmd = 0.1 if overall_ratio < 2.0 else (0.2 if overall_ratio < 4.0 else 0.3)'
    new_lambda = f'lambda_mmd = {{lambda1}} if overall_ratio < {{thresh1}} else ({{lambda2}} if overall_ratio < {{thresh2}} else {{lambda3}})'
    modified_content = modified_content.replace(old_lambda, new_lambda)

    small_w_decay, small_mmd_decay, large_mmd_decay = params['guard_factors']
    modified_content = modified_content.replace(
        'new_w = max(1.0, cur_w * 0.8)',
        f'new_w = max(1.0, cur_w * {{small_w_decay}})'
    )
    modified_content = modified_content.replace(
        'new_lambda = max(0.0, cur_lambda * 0.5)',
        f'new_lambda = max(0.0, cur_lambda * {{small_mmd_decay}})'
    )
    modified_content = modified_content.replace(
        'new_lambda = max(0.0, cur_lambda * 0.7)',
        f'new_lambda = max(0.0, cur_lambda * {{large_mmd_decay}})'
    )

    min_epochs, max_epochs, ratio = params['warmup_config']
    old_warmup = 'warmup = max(2, min(5, int(0.1 * MAX_EPOCHS)))'
    new_warmup = f'warmup = max({{min_epochs}}, min({{max_epochs}}, int({{ratio}} * MAX_EPOCHS)))'
    modified_content = modified_content.replace(old_warmup, new_warmup)

    return modified_content


def run_final_experiment(focus_case: str, optimal_result: dict) -> dict:
    """Run final experiment with optimal parameters"""
    print(f"\\n{{'='*60}}")
    print(f"FINAL EXPERIMENT: {{focus_case}}")
    print(f"Expected accuracy: {{optimal_result['accuracy']:.4f}}")
    print(f"{{'='*60}}")

    params = optimal_result['parameters']

    # Configuration based on focus case
    if focus_case == 'P3_small':
        config_overrides = {{
            'use_combined_datasets': True,
            'data_dir': 'P3_DATA_DIR',
            'dataset': 'use_combined_datasets',
            'NESTED_CV_TRIALS_PER_SUBJECT_P3': 20,   # Small P3
            'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 200, # Large AVO
            'NESTED_CV_OUTER_FOLDS': 5,              # Full CV
            'NESTED_CV_REPEATS': 5,
            'EARLY_STOPPING_PATIENCE': params['early_stop_patience'],
            'LEARNING_RATE': params['learning_rate'],
        }}
    else:  # AVO_small
        config_overrides = {{
            'use_combined_datasets': True,
            'data_dir': 'P3_DATA_DIR',
            'dataset': 'use_combined_datasets',
            'NESTED_CV_TRIALS_PER_SUBJECT_P3': 200,  # Large P3
            'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 20,  # Small AVO
            'NESTED_CV_OUTER_FOLDS': 5,
            'NESTED_CV_REPEATS': 5,
            'EARLY_STOPPING_PATIENCE': params['early_stop_patience'],
            'LEARNING_RATE': params['learning_rate'],
        }}

    # Create temporary directory
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"final_{{focus_case}}_"))

    try:
        # Create modified config
        base_config = (PROJECT_ROOT / 'config.py').read_text(encoding='utf-8')
        modified_config = create_modified_config(base_config, config_overrides)
        (tmp_dir / 'config.py').write_text(modified_config, encoding='utf-8')

        # Create modified main_tfdwt.py
        modified_tfdwt = create_modified_tfdwt(params)
        (tmp_dir / 'main_tfdwt.py').write_text(modified_tfdwt, encoding='utf-8')

        # Setup environment
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{{str(tmp_dir)}}:{{str(PROJECT_ROOT)}}"
        env['CONFIG_OVERRIDE_PATH'] = str(tmp_dir / 'config.py')

        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"final_optimized_{{focus_case}}_{{timestamp}}.log"
        log_path = FINAL_LOG_DIR / log_filename

        python_code = f"""
import sys
import os
sys.path.insert(0, '{{str(tmp_dir)}}')
os.environ['CONFIG_OVERRIDE_PATH'] = '{{str(tmp_dir / "config.py")}}'
import runpy
runpy.run_path('{{str(tmp_dir / "main_tfdwt.py")}}', run_name='__main__')
"""

        cmd = [sys.executable, '-c', python_code]

        print("Starting final optimized experiment...")
        print("Parameters:")
        for key, value in params.items():
            print(f"  {{key}}: {{value}}")

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
                        'fold', 'accuracy', 'completed', 'final results', 'saved to'
                    ]):
                        print(f"[{{focus_case}}] {{line}}")

            proc.wait()

        # Extract final accuracy
        final_accuracy = 0.0
        csv_files = []

        for line in output_lines:
            if 'Final Results: Overall Accuracy' in line:
                import re
                match = re.search(r'Overall Accuracy = ([\\d.]+)', line)
                if match:
                    final_accuracy = float(match.group(1))

            # Look for CSV files
            if 'detailed results saved to:' in line.lower() or 'tfdwt_detailed_results_' in line:
                import re
                csv_match = re.search(r'tfdwt_detailed_results_\\d+_\\d+\\.csv', line)
                if csv_match:
                    csv_file = PROJECT_ROOT / csv_match.group(0)
                    if csv_file.exists():
                        csv_files.append(csv_file)

        result = {{
            'focus_case': focus_case,
            'final_accuracy': final_accuracy,
            'expected_accuracy': optimal_result['accuracy'],
            'improvement': final_accuracy - optimal_result['accuracy'],
            'optimal_parameters': params,
            'log_file': str(log_path),
            'csv_files': [str(f) for f in csv_files],
            'return_code': proc.returncode,
        }}

        print(f"\\n🎯 {{focus_case}} FINAL RESULT:")
        print(f"  Final accuracy: {{final_accuracy:.4f}}")
        print(f"  Expected: {{optimal_result['accuracy']:.4f}}")
        print(f"  Improvement: {{result['improvement']:+.4f}}")
        print(f"  Log: {{log_path.name}}")

        return result

    except Exception as e:
        print(f"❌ Final experiment failed: {{e}}")
        return {{
            'focus_case': focus_case,
            'final_accuracy': 0.0,
            'error': str(e),
            'return_code': -1
        }}
    finally:
        # Clean up
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    """Run final optimized experiments"""
    print("🎯 Final TF-DWT Experiments with Optimized Parameters")

    final_results = []

    for focus_case in ['P3_small', 'AVO_small']:
        optimal_result = OPTIMAL_PARAMS[focus_case]
        result = run_final_experiment(focus_case, optimal_result)
        final_results.append(result)

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = FINAL_LOG_DIR / f"final_optimization_results_{{timestamp}}.json"

    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\\n📁 Final results saved to: {{results_file}}")

    # Summary
    print(f"\\n{{'='*60}}")
    print("FINAL EXPERIMENT SUMMARY")
    print(f"{{'='*60}}")

    for result in final_results:
        focus_case = result['focus_case']
        final_acc = result.get('final_accuracy', 0)
        improvement = result.get('improvement', 0)

        if final_acc > 0:
            print(f"✅ {{focus_case}}: {{final_acc:.4f}} ({{improvement:+.4f}})")
        else:
            print(f"❌ {{focus_case}}: Failed")

    return final_results


if __name__ == '__main__':
    main()
'''

    final_script_path = PROJECT_ROOT / 'run_final_optimized_experiments.py'
    with open(final_script_path, 'w') as f:
        f.write(script_content)

    # Make executable
    import os
    os.chmod(final_script_path, 0o755)

    return final_script_path


def print_analysis_summary(analysis: Dict) -> None:
    """Print detailed analysis summary"""
    focus_case = analysis['focus_case']

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION ANALYSIS: {focus_case}")
    print(f"{'='*60}")

    # Stage-by-stage results
    for stage, stage_data in analysis['stages'].items():
        print(f"\n📊 {stage.upper()} Stage:")
        print(f"  Experiments: {stage_data['successful_experiments']}/{stage_data['total_experiments']} successful")
        print(f"  Success rate: {stage_data['success_rate']:.1%}")
        print(f"  Best accuracy: {stage_data['best_accuracy']:.4f}")
        print(f"  Mean accuracy: {stage_data['mean_accuracy']:.4f} ± {stage_data['std_accuracy']:.4f}")

        if 'cv_config' in stage_data:
            cv = stage_data['cv_config']
            print(f"  CV config: {cv.get('folds', '?')} folds, {cv.get('repeats', '?')} repeats")

    # Overall best
    if analysis['best_overall']:
        best = analysis['best_overall']
        print(f"\n🏆 OVERALL BEST:")
        print(f"  Accuracy: {best['accuracy']:.4f}")
        print(f"  Stage: {best['experiment_id'].split('_')[0]}")
        print(f"  Parameters:")
        for key, value in best['parameters'].items():
            print(f"    {key}: {value}")

    # Parameter trends
    print(f"\n📈 PARAMETER TRENDS:")
    trends = analysis['parameter_trends']

    for param_name, param_values in trends.items():
        if len(param_values) > 1:  # Only show if there are multiple values
            print(f"\n  {param_name}:")

            # Sort by mean accuracy
            sorted_values = sorted(param_values.items(),
                                 key=lambda x: x[1]['mean_accuracy'],
                                 reverse=True)

            for value, stats in sorted_values[:3]:  # Show top 3
                print(f"    {value}: {stats['mean_accuracy']:.4f} "
                      f"(max: {stats['max_accuracy']:.4f}, n={stats['count']})")


def main():
    """Main analysis function"""
    print("📊 TF-DWT Optimization Results Analysis")

    results = {}

    # Load and analyze results for both cases
    for focus_case in ['P3_small', 'AVO_small']:
        print(f"\nAnalyzing {focus_case} results...")

        data = load_optimization_results(focus_case)
        if data:
            analysis = analyze_stage_results(data['results_by_stage'], focus_case)
            results[focus_case] = analysis

            print_analysis_summary(analysis)
        else:
            results[focus_case] = None

    # Create final experiment script if both analyses succeeded
    if results['P3_small'] and results['AVO_small']:
        p3_best = results['P3_small']['best_overall']
        avo_best = results['AVO_small']['best_overall']

        if p3_best and avo_best:
            print(f"\n🚀 Creating final experiment script...")
            final_script = create_final_experiment_script(p3_best, avo_best)
            print(f"📁 Final experiment script created: {final_script}")

            print(f"\n📋 To run final optimized experiments:")
            print(f"  python {final_script.name}")
        else:
            print(f"\n⚠️  Cannot create final script - missing best results")
    else:
        print(f"\n⚠️  Cannot create final script - incomplete optimization results")

    # Save analysis results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = RESULTS_DIR / f"optimization_analysis_{timestamp}.json"

    with open(analysis_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📁 Analysis saved to: {analysis_file}")

    return results


if __name__ == '__main__':
    main()