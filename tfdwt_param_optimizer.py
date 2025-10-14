"""
TF-DWT Parameter Optimization Script

This script systematically optimizes TF-DWT algorithm parameters to improve
performance on small datasets. It uses a staged approach:

1. Quick screening: Fast parameter combinations with reduced CV
2. Full evaluation: Complete nested CV with best candidates
3. Result analysis: Statistical comparison and optimal parameter selection

Key parameters optimized:
- w_small_target upper limit (domain weight cap)
- MMD lambda values and thresholds (alignment strength)
- Guard decay factors (stability mechanisms)
- Warmup parameters (training schedule)

Usage:
  python tfdwt_param_optimizer.py --stage [screening|full] --focus [P3_small|AVO_small]
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from itertools import product

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_CONFIG_PATH = PROJECT_ROOT / 'config.py'

# Create optimization log directory
OPTIM_LOG_DIR = PROJECT_ROOT / 'log_tfdwt_optimization'
OPTIM_LOG_DIR.mkdir(exist_ok=True)

# Parameter ranges for optimization
PARAM_RANGES = {
    # High impact parameters
    'w_small_cap': [2.0, 2.5, 3.0, 3.5, 4.0],  # Weight cap for small domain
    'mmd_thresholds': [
        (1.5, 3.0, 0.05, 0.15, 0.25),  # (thresh1, thresh2, lambda1, lambda2, lambda3)
        (2.0, 4.0, 0.1, 0.2, 0.3),     # Current default
        (2.5, 5.0, 0.15, 0.25, 0.35),
    ],

    # Medium impact parameters
    'guard_factors': [
        (0.7, 0.4, 0.6),  # (small_w_decay, small_mmd_decay, large_mmd_decay) - more aggressive
        (0.8, 0.5, 0.7),  # Current default
        (0.9, 0.6, 0.8),  # More conservative
    ],
    'warmup_config': [
        (2, 5, 0.08),  # (min_epochs, max_epochs, ratio) - shorter warmup
        (2, 5, 0.1),   # Current default
        (3, 7, 0.12),  # longer warmup
    ],

    # Lower impact but still important
    'early_stop_patience': [30, 40, 50, 60],
    'learning_rate': [0.008, 0.01, 0.012, 0.015],
}

# Quick screening parameter combinations (reduced set)
QUICK_PARAM_RANGES = {
    'w_small_cap': [2.5, 3.0, 3.5],
    'mmd_thresholds': [
        (1.5, 3.0, 0.05, 0.15, 0.25),
        (2.0, 4.0, 0.1, 0.2, 0.3),
        (2.5, 5.0, 0.15, 0.25, 0.35),
    ],
    'guard_factors': [
        (0.8, 0.5, 0.7),  # Current default only
    ],
    'warmup_config': [
        (2, 5, 0.1),   # Current default only
    ],
    'early_stop_patience': [40, 50],
    'learning_rate': [0.01, 0.012],
}


class TFDWTOptimizer:
    def __init__(self, focus_case: str, stage: str = 'screening'):
        self.focus_case = focus_case  # 'P3_small' or 'AVO_small'
        self.stage = stage
        self.results = []
        self.best_params = None
        self.best_accuracy = 0.0

        # Configure experiment based on focus
        if focus_case == 'P3_small':
            self.base_config = {
                'use_combined_datasets': True,
                'data_dir': 'P3_DATA_DIR',
                'dataset': 'use_combined_datasets',
                'NESTED_CV_TRIALS_PER_SUBJECT_P3': 20,   # Small P3
                'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 200, # Large AVO
            }
        elif focus_case == 'AVO_small':
            self.base_config = {
                'use_combined_datasets': True,
                'data_dir': 'P3_DATA_DIR',
                'dataset': 'use_combined_datasets',
                'NESTED_CV_TRIALS_PER_SUBJECT_P3': 200,  # Large P3
                'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 20,  # Small AVO
            }
        else:
            raise ValueError("focus_case must be 'P3_small' or 'AVO_small'")

        # Adjust CV for screening vs full evaluation
        if stage == 'screening':
            # Reduced CV for faster screening
            self.base_config.update({
                'NESTED_CV_OUTER_FOLDS': 3,
                'NESTED_CV_REPEATS': 2,
            })
            self.param_ranges = QUICK_PARAM_RANGES
        else:
            # Full CV for final evaluation
            self.base_config.update({
                'NESTED_CV_OUTER_FOLDS': 5,
                'NESTED_CV_REPEATS': 5,
            })
            self.param_ranges = PARAM_RANGES

    def read_base_config(self) -> str:
        return BASE_CONFIG_PATH.read_text(encoding='utf-8')

    def apply_config_overrides(self, config_text: str, overrides: Dict[str, Any]) -> str:
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

        # Apply base configuration overrides
        for key, value in overrides.items():
            if key in ['use_combined_datasets']:
                set_line(key, 'True' if value else 'False')
            elif key in ['data_dir']:
                set_line(key, str(value))
            elif key in ['dataset']:
                set_line(key, repr(value))
            elif key.startswith('NESTED_CV'):
                set_line(key, str(value))

        return "\n".join(lines) + "\n"

    def create_modified_tfdwt(self, params: Dict[str, Any]) -> str:
        """Create a modified version of main_tfdwt.py with optimized parameters"""
        tfdwt_path = PROJECT_ROOT / 'main_tfdwt.py'
        original_content = tfdwt_path.read_text(encoding='utf-8')

        # Apply parameter modifications
        modified_content = original_content

        # 1. Modify w_small_target cap (line ~320)
        w_cap = params['w_small_cap']
        modified_content = modified_content.replace(
            'w_small_target = min(w_small_target, 3.0)',
            f'w_small_target = min(w_small_target, {w_cap})'
        )

        # 2. Modify MMD lambda thresholds and values (lines ~208-209)
        thresh1, thresh2, lambda1, lambda2, lambda3 = params['mmd_thresholds']
        old_lambda_logic = 'lambda_mmd = 0.1 if overall_ratio < 2.0 else (0.2 if overall_ratio < 4.0 else 0.3)'
        new_lambda_logic = f'lambda_mmd = {lambda1} if overall_ratio < {thresh1} else ({lambda2} if overall_ratio < {thresh2} else {lambda3})'
        modified_content = modified_content.replace(old_lambda_logic, new_lambda_logic)

        # 3. Modify guard decay factors (lines ~293-294, ~302)
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

        # 4. Modify warmup parameters (line ~210)
        min_epochs, max_epochs, ratio = params['warmup_config']
        old_warmup = f'warmup = max(2, min(5, int(0.1 * MAX_EPOCHS)))'
        new_warmup = f'warmup = max({min_epochs}, min({max_epochs}, int({ratio} * MAX_EPOCHS)))'
        modified_content = modified_content.replace(old_warmup, new_warmup)

        return modified_content

    def run_experiment(self, params: Dict[str, Any], experiment_id: str) -> Dict[str, float]:
        """Run a single experiment with given parameters"""

        # Create temporary directories
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"tfdwt_optim_{experiment_id}_"))

        try:
            # Create modified config
            base_config = self.read_base_config()
            config_overrides = self.base_config.copy()
            config_overrides.update({
                'EARLY_STOPPING_PATIENCE': params['early_stop_patience'],
                'LEARNING_RATE': params['learning_rate'],
            })

            modified_config = self.apply_config_overrides(base_config, config_overrides)
            (tmp_dir / 'config.py').write_text(modified_config, encoding='utf-8')

            # Create modified main_tfdwt.py
            modified_tfdwt = self.create_modified_tfdwt(params)
            (tmp_dir / 'main_tfdwt.py').write_text(modified_tfdwt, encoding='utf-8')

            # Prepare execution environment
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{str(tmp_dir)}:{str(PROJECT_ROOT)}"
            env['CONFIG_OVERRIDE_PATH'] = str(tmp_dir / 'config.py')

            # Create log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{self.focus_case}_{self.stage}_{experiment_id}_{timestamp}.log"
            log_path = OPTIM_LOG_DIR / log_filename

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

            print(f"[{experiment_id}] Starting experiment with parameters:")
            for key, value in params.items():
                print(f"  {key}: {value}")

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

                        # Print key progress
                        line = output.strip()
                        if any(keyword in line.lower() for keyword in [
                            'fold', 'accuracy', 'completed', 'error', 'failed', 'final results'
                        ]):
                            print(f"[{experiment_id}] {line}")

                proc.wait()

            # Extract results from output
            results = self.extract_results_from_output(output_lines, log_path)
            results['experiment_id'] = experiment_id
            results['log_file'] = str(log_path)
            results['return_code'] = proc.returncode
            results['parameters'] = params.copy()

            return results

        except Exception as e:
            print(f"[{experiment_id}] ERROR: {e}")
            return {
                'experiment_id': experiment_id,
                'overall_accuracy': 0.0,
                'p3_accuracy': 0.0,
                'avo_accuracy': 0.0,
                'return_code': -1,
                'error': str(e),
                'parameters': params.copy()
            }
        finally:
            # Clean up temporary directory
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def extract_results_from_output(self, output_lines: List[str], log_path: Path) -> Dict[str, float]:
        """Extract accuracy results from experiment output"""
        results = {
            'overall_accuracy': 0.0,
            'p3_accuracy': 0.0,
            'avo_accuracy': 0.0,
            'overall_ci_lower': 0.0,
            'overall_ci_upper': 0.0,
        }

        try:
            # Look for final results in output
            for line in output_lines:
                if 'Final Results: Overall Accuracy' in line:
                    # Extract: Final Results: Overall Accuracy = 0.7234
                    import re
                    match = re.search(r'Overall Accuracy = ([\d.]+)', line)
                    if match:
                        results['overall_accuracy'] = float(match.group(1))

            # Also try to extract from detailed CSV files if they exist
            # Look for CSV file mentions in output
            csv_files = []
            for line in output_lines:
                if 'detailed results saved to:' in line.lower() or 'tfdwt_detailed_results_' in line:
                    import re
                    csv_match = re.search(r'tfdwt_detailed_results_\d+_\d+\.csv', line)
                    if csv_match:
                        csv_file = PROJECT_ROOT / csv_match.group(0)
                        if csv_file.exists():
                            csv_files.append(csv_file)

            # Extract detailed metrics from CSV if available
            if csv_files:
                df = pd.read_csv(csv_files[-1])  # Use the most recent one
                if not df.empty:
                    results['overall_accuracy'] = df['overall_accuracy'].mean()
                    results['p3_accuracy'] = df['p3_accuracy'].mean()
                    results['avo_accuracy'] = df['avo_accuracy'].mean()

                    # Calculate confidence intervals
                    n = len(df)
                    if n > 1:
                        from scipy import stats
                        mean_acc = results['overall_accuracy']
                        std_acc = df['overall_accuracy'].std(ddof=1)
                        t_crit = stats.t.ppf(0.975, df=n-1)  # 95% CI
                        margin = t_crit * (std_acc / np.sqrt(n))
                        results['overall_ci_lower'] = mean_acc - margin
                        results['overall_ci_upper'] = mean_acc + margin

        except Exception as e:
            print(f"Warning: Could not extract detailed results: {e}")

        return results

    def generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for testing"""
        param_names = list(self.param_ranges.keys())
        param_values = [self.param_ranges[name] for name in param_names]

        combinations = []
        for combo in product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)

        return combinations

    def run_optimization(self) -> None:
        """Run the full parameter optimization process"""
        print(f"Starting TF-DWT {self.stage} optimization for {self.focus_case}")
        print(f"Results will be saved to: {OPTIM_LOG_DIR}")

        param_combinations = self.generate_param_combinations()
        print(f"Total parameter combinations to test: {len(param_combinations)}")

        if self.stage == 'screening' and len(param_combinations) > 20:
            print("Screening stage: Testing first 20 combinations for quick evaluation")
            param_combinations = param_combinations[:20]

        # Run experiments
        for i, params in enumerate(param_combinations, 1):
            experiment_id = f"{self.stage}_{i:03d}"
            print(f"\n{'='*60}")
            print(f"Running experiment {i}/{len(param_combinations)}: {experiment_id}")
            print(f"{'='*60}")

            results = self.run_experiment(params, experiment_id)
            self.results.append(results)

            # Update best results
            current_acc = results.get('overall_accuracy', 0.0)
            if current_acc > self.best_accuracy:
                self.best_accuracy = current_acc
                self.best_params = params.copy()
                print(f"🏆 New best accuracy: {current_acc:.4f}")

            # Save intermediate results
            self.save_results()

            print(f"[{experiment_id}] Completed - Accuracy: {current_acc:.4f}")

        # Final summary
        self.print_optimization_summary()

    def save_results(self) -> None:
        """Save optimization results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results to JSON
        results_file = OPTIM_LOG_DIR / f"tfdwt_optimization_{self.focus_case}_{self.stage}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'focus_case': self.focus_case,
                'stage': self.stage,
                'best_accuracy': self.best_accuracy,
                'best_params': self.best_params,
                'all_results': self.results
            }, f, indent=2)

        # Save results to CSV for analysis
        if self.results:
            df_rows = []
            for result in self.results:
                row = {
                    'experiment_id': result['experiment_id'],
                    'overall_accuracy': result.get('overall_accuracy', 0.0),
                    'p3_accuracy': result.get('p3_accuracy', 0.0),
                    'avo_accuracy': result.get('avo_accuracy', 0.0),
                    'return_code': result.get('return_code', -1),
                }
                # Add parameter values
                if 'parameters' in result:
                    for param_name, param_value in result['parameters'].items():
                        if isinstance(param_value, (list, tuple)):
                            row[f'param_{param_name}'] = str(param_value)
                        else:
                            row[f'param_{param_name}'] = param_value
                df_rows.append(row)

            df = pd.DataFrame(df_rows)
            csv_file = OPTIM_LOG_DIR / f"tfdwt_optimization_{self.focus_case}_{self.stage}_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            print(f"Results saved to: {csv_file}")

    def print_optimization_summary(self) -> None:
        """Print optimization summary"""
        print(f"\n{'='*80}")
        print("OPTIMIZATION SUMMARY")
        print(f"{'='*80}")
        print(f"Focus case: {self.focus_case}")
        print(f"Stage: {self.stage}")
        print(f"Total experiments: {len(self.results)}")

        if self.best_params:
            print(f"\n🏆 BEST ACCURACY: {self.best_accuracy:.4f}")
            print("📋 OPTIMAL PARAMETERS:")
            for param_name, param_value in self.best_params.items():
                print(f"  {param_name}: {param_value}")

        # Show top 5 results
        sorted_results = sorted(self.results,
                              key=lambda x: x.get('overall_accuracy', 0.0),
                              reverse=True)[:5]

        print(f"\n📊 TOP 5 RESULTS:")
        for i, result in enumerate(sorted_results, 1):
            acc = result.get('overall_accuracy', 0.0)
            exp_id = result.get('experiment_id', 'unknown')
            print(f"  {i}. {exp_id}: {acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description='TF-DWT Parameter Optimization')
    parser.add_argument('--stage', choices=['screening', 'full'], default='screening',
                       help='Optimization stage: screening (fast) or full (complete)')
    parser.add_argument('--focus', choices=['P3_small', 'AVO_small'], required=True,
                       help='Focus case: P3_small or AVO_small')

    args = parser.parse_args()

    optimizer = TFDWTOptimizer(focus_case=args.focus, stage=args.stage)
    optimizer.run_optimization()


if __name__ == '__main__':
    main()