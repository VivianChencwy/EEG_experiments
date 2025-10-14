#!/usr/bin/env python3
"""
Staged TF-DWT Parameter Optimization

This script implements a practical parameter optimization strategy:

Stage 1: Rapid screening with minimal CV (1 fold, 1 repeat)
Stage 2: Validation with reduced CV (2 folds, 2 repeats)
Stage 3: Final evaluation with full CV (5 folds, 5 repeats)

The goal is to quickly identify promising parameter combinations before
investing time in full cross-validation.
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

PROJECT_ROOT = Path(__file__).resolve().parent

# Create optimization log directory
OPTIM_LOG_DIR = PROJECT_ROOT / 'log_staged_optimization'
OPTIM_LOG_DIR.mkdir(exist_ok=True)

# Parameter ranges for different stages
STAGE_CONFIGS = {
    'rapid': {
        'cv_folds': 2,
        'cv_repeats': 1,
        'max_epochs_override': 100,  # Faster training
        'params': {
            # Test key parameters only
            'w_small_cap': [2.5, 3.0, 3.5],
            'mmd_thresholds': [
                (2.0, 4.0, 0.1, 0.2, 0.3),     # Current default
                (1.5, 3.0, 0.05, 0.15, 0.25),  # More conservative
            ],
            'guard_factors': [
                (0.8, 0.5, 0.7),  # Current default
            ],
            'warmup_config': [
                (2, 5, 0.1),   # Current default
            ],
            'early_stop_patience': [20],
            'learning_rate': [0.01, 0.012],
        }
    },
    'validation': {
        'cv_folds': 3,
        'cv_repeats': 2,
        'max_epochs_override': 200,
        'params': {
            # Expand best parameters from rapid stage
            'w_small_cap': [2.0, 2.5, 3.0, 3.5, 4.0],
            'mmd_thresholds': [
                (1.5, 3.0, 0.05, 0.15, 0.25),
                (2.0, 4.0, 0.1, 0.2, 0.3),
                (2.5, 5.0, 0.15, 0.25, 0.35),
            ],
            'guard_factors': [
                (0.7, 0.4, 0.6),  # More aggressive
                (0.8, 0.5, 0.7),  # Current default
                (0.9, 0.6, 0.8),  # More conservative
            ],
            'warmup_config': [
                (2, 5, 0.08),
                (2, 5, 0.1),
                (3, 7, 0.12),
            ],
            'early_stop_patience': [15, 20, 25],
            'learning_rate': [0.008, 0.01, 0.012, 0.015],
        }
    },
    'final': {
        'cv_folds': 5,
        'cv_repeats': 5,
        'max_epochs_override': None,  # Use config default
        'params': None,  # Will be set based on validation results
    }
}


class StagedOptimizer:
    def __init__(self, focus_case: str):
        self.focus_case = focus_case  # 'P3_small' or 'AVO_small'
        self.results_by_stage = {}
        self.best_params_by_stage = {}

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

    def generate_param_combinations(self, param_ranges: Dict) -> List[Dict]:
        """Generate parameter combinations"""
        from itertools import product

        param_names = list(param_ranges.keys())
        param_values = [param_ranges[name] for name in param_names]

        combinations = []
        for combo in product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)

        return combinations

    def select_top_params(self, results: List[Dict], top_k: int = 3) -> List[Dict]:
        """Select top performing parameter combinations"""
        # Filter successful results
        valid_results = [r for r in results if r.get('return_code') == 0 and r.get('accuracy', 0) > 0]

        if not valid_results:
            print("⚠️  No valid results to select from")
            return []

        # Sort by accuracy
        sorted_results = sorted(valid_results, key=lambda x: x.get('accuracy', 0), reverse=True)

        # Extract parameter sets from top results
        top_params = []
        for result in sorted_results[:top_k]:
            if 'parameters' in result:
                top_params.append(result['parameters'])

        return top_params

    def expand_around_best(self, best_params: List[Dict]) -> Dict:
        """Create expanded parameter ranges around best performing parameters"""
        if not best_params:
            return STAGE_CONFIGS['validation']['params']  # Fallback to full range

        # Extract values from best parameters
        w_caps = [p['w_small_cap'] for p in best_params]
        learning_rates = [p['learning_rate'] for p in best_params]

        # Create expanded ranges
        expanded = {
            'w_small_cap': list(set([
                max(1.5, min(w) - 0.5),
                min(w),
                max(w),
                min(5.0, max(w) + 0.5)
            ] for w in [w_caps]))[0],

            'mmd_thresholds': list(set(p['mmd_thresholds'] for p in best_params)),
            'guard_factors': list(set(p['guard_factors'] for p in best_params)),
            'warmup_config': list(set(p['warmup_config'] for p in best_params)),

            'early_stop_patience': [15, 20, 25, 30],
            'learning_rate': list(set([
                max(0.005, min(learning_rates) - 0.002),
                *learning_rates,
                min(0.02, max(learning_rates) + 0.002)
            ])),
        }

        return expanded

    def run_stage(self, stage: str, param_ranges: Dict = None) -> List[Dict]:
        """Run optimization for a specific stage"""
        print(f"\n{'='*60}")
        print(f"STAGE: {stage.upper()} OPTIMIZATION")
        print(f"{'='*60}")

        stage_config = STAGE_CONFIGS[stage]

        if param_ranges is None:
            param_ranges = stage_config['params']

        if param_ranges is None:
            raise ValueError(f"No parameter ranges specified for stage {stage}")

        # Generate parameter combinations
        combinations = self.generate_param_combinations(param_ranges)
        print(f"Testing {len(combinations)} parameter combinations")

        # Limit combinations for rapid stage
        if stage == 'rapid' and len(combinations) > 12:
            combinations = combinations[:12]
            print(f"Limited to {len(combinations)} combinations for rapid screening")

        results = []
        best_accuracy = 0.0

        for i, params in enumerate(combinations, 1):
            experiment_id = f"{stage}_{i:02d}"
            print(f"\n[{experiment_id}] Running experiment {i}/{len(combinations)}")

            # Run single experiment
            result = self.run_single_experiment(
                params, experiment_id, stage_config
            )

            results.append(result)

            # Track best
            current_acc = result.get('accuracy', 0.0)
            if current_acc > best_accuracy:
                best_accuracy = current_acc
                print(f"🏆 New best accuracy: {current_acc:.4f}")

        # Save stage results
        self.results_by_stage[stage] = results

        # Select best parameters for next stage
        if stage != 'final':
            top_params = self.select_top_params(results, top_k=3)
            self.best_params_by_stage[stage] = top_params

            print(f"\n📊 Stage {stage} complete:")
            print(f"  Best accuracy: {best_accuracy:.4f}")
            print(f"  Top parameter sets: {len(top_params)}")

        return results

    def run_single_experiment(self, params: Dict, experiment_id: str, stage_config: Dict) -> Dict:
        """Run a single experiment with given parameters"""

        # Create temporary directory
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"staged_{experiment_id}_"))

        try:
            # Create config overrides
            config_overrides = self.base_config.copy()
            config_overrides.update({
                'NESTED_CV_OUTER_FOLDS': stage_config['cv_folds'],
                'NESTED_CV_REPEATS': stage_config['cv_repeats'],
                'EARLY_STOPPING_PATIENCE': params['early_stop_patience'],
                'LEARNING_RATE': params['learning_rate'],
            })

            # Add max epochs override for faster stages
            if stage_config.get('max_epochs_override'):
                config_overrides['MAX_EPOCHS'] = stage_config['max_epochs_override']

            # Create modified config
            modified_config = self.create_modified_config(config_overrides)
            (tmp_dir / 'config.py').write_text(modified_config, encoding='utf-8')

            # Create modified main_tfdwt.py
            modified_tfdwt = self.create_modified_tfdwt(params)
            (tmp_dir / 'main_tfdwt.py').write_text(modified_tfdwt, encoding='utf-8')

            # Run experiment
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

            # Print experiment info
            print(f"  Parameters: w_cap={params['w_small_cap']}, lr={params['learning_rate']}")

            # Run with output capture
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
                    # Print progress
                    line = output.strip()
                    if any(keyword in line.lower() for keyword in [
                        'final results', 'accuracy', 'completed'
                    ]):
                        print(f"    {line}")

            # Extract accuracy
            accuracy = self.extract_accuracy_from_output(output_lines)

            return {
                'experiment_id': experiment_id,
                'accuracy': accuracy,
                'parameters': params.copy(),
                'return_code': proc.returncode,
                'cv_config': {
                    'folds': stage_config['cv_folds'],
                    'repeats': stage_config['cv_repeats']
                }
            }

        except Exception as e:
            print(f"    ERROR: {e}")
            return {
                'experiment_id': experiment_id,
                'accuracy': 0.0,
                'parameters': params.copy(),
                'return_code': -1,
                'error': str(e)
            }
        finally:
            # Cleanup
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def create_modified_config(self, overrides: Dict) -> str:
        """Create modified config.py content"""
        base_config = (PROJECT_ROOT / 'config.py').read_text(encoding='utf-8')

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

        for key, value in overrides.items():
            if key in ['use_combined_datasets']:
                set_line(key, 'True' if value else 'False')
            elif key in ['data_dir']:
                set_line(key, str(value))
            elif key in ['dataset']:
                set_line(key, repr(value))
            else:
                set_line(key, str(value))

        return "\n".join(lines) + "\n"

    def create_modified_tfdwt(self, params: Dict) -> str:
        """Create modified main_tfdwt.py content"""
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

    def extract_accuracy_from_output(self, output_lines: List[str]) -> float:
        """Extract accuracy from experiment output"""
        for line in output_lines:
            if 'Final Results: Overall Accuracy' in line:
                import re
                match = re.search(r'Overall Accuracy = ([\d.]+)', line)
                if match:
                    return float(match.group(1))
        return 0.0

    def run_full_optimization(self):
        """Run the complete staged optimization process"""
        print(f"🚀 Starting Staged TF-DWT Optimization for {self.focus_case}")

        start_time = datetime.now()

        # Stage 1: Rapid screening
        print("\n" + "="*80)
        print("STAGE 1: RAPID SCREENING")
        print("="*80)

        self.run_stage('rapid')

        # Stage 2: Validation with expanded ranges
        if self.best_params_by_stage.get('rapid'):
            print("\n" + "="*80)
            print("STAGE 2: VALIDATION")
            print("="*80)

            # Expand parameter ranges around best from rapid stage
            expanded_params = self.expand_around_best(self.best_params_by_stage['rapid'])
            self.run_stage('validation', expanded_params)
        else:
            print("\n⚠️  Skipping validation stage - no good rapid results")

        # Stage 3: Final evaluation
        if self.best_params_by_stage.get('validation'):
            print("\n" + "="*80)
            print("STAGE 3: FINAL EVALUATION")
            print("="*80)

            # Use only the very best parameter combination
            final_params = self.best_params_by_stage['validation'][:1]  # Just the best one
            final_param_ranges = {}

            if final_params:
                # Create single-value ranges for the best parameters
                best = final_params[0]
                for key, value in best.items():
                    final_param_ranges[key] = [value]  # Single value range

                STAGE_CONFIGS['final']['params'] = final_param_ranges
                self.run_stage('final')
            else:
                print("⚠️  No validation results for final stage")
        else:
            print("\n⚠️  Skipping final stage - no good validation results")

        # Generate summary
        self.generate_summary(start_time)

    def generate_summary(self, start_time: datetime):
        """Generate optimization summary"""
        end_time = datetime.now()
        duration = end_time - start_time

        print(f"\n{'='*80}")
        print("STAGED OPTIMIZATION SUMMARY")
        print(f"{'='*80}")

        print(f"Focus case: {self.focus_case}")
        print(f"Total time: {duration}")

        # Results by stage
        for stage in ['rapid', 'validation', 'final']:
            if stage in self.results_by_stage:
                results = self.results_by_stage[stage]
                valid_results = [r for r in results if r.get('accuracy', 0) > 0]

                if valid_results:
                    best_result = max(valid_results, key=lambda x: x['accuracy'])
                    print(f"\n📊 {stage.upper()} stage:")
                    print(f"  Best accuracy: {best_result['accuracy']:.4f}")
                    print(f"  Total experiments: {len(results)}")
                    print(f"  Success rate: {len(valid_results)}/{len(results)}")
                else:
                    print(f"\n❌ {stage.upper()} stage: No valid results")

        # Save detailed results
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")

        summary_data = {
            'focus_case': self.focus_case,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'results_by_stage': self.results_by_stage,
            'best_params_by_stage': self.best_params_by_stage,
        }

        # Find overall best result
        all_results = []
        for stage_results in self.results_by_stage.values():
            all_results.extend(stage_results)

        valid_results = [r for r in all_results if r.get('accuracy', 0) > 0]
        if valid_results:
            overall_best = max(valid_results, key=lambda x: x['accuracy'])
            summary_data['overall_best'] = overall_best

            print(f"\n🏆 OVERALL BEST RESULT:")
            print(f"  Accuracy: {overall_best['accuracy']:.4f}")
            print(f"  Stage: {overall_best['experiment_id'].split('_')[0]}")
            print(f"  Parameters: {overall_best['parameters']}")

        # Save to files
        summary_file = OPTIM_LOG_DIR / f"staged_optimization_{self.focus_case}_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)

        # Create CSV for analysis
        if all_results:
            df_rows = []
            for result in all_results:
                row = {
                    'experiment_id': result['experiment_id'],
                    'stage': result['experiment_id'].split('_')[0],
                    'accuracy': result.get('accuracy', 0.0),
                    'return_code': result.get('return_code', -1),
                }
                if 'parameters' in result:
                    for key, value in result['parameters'].items():
                        row[f'param_{key}'] = str(value) if isinstance(value, (list, tuple)) else value
                df_rows.append(row)

            df = pd.DataFrame(df_rows)
            csv_file = OPTIM_LOG_DIR / f"staged_optimization_{self.focus_case}_{timestamp}.csv"
            df.to_csv(csv_file, index=False)

            print(f"\n📁 Results saved:")
            print(f"  Summary: {summary_file}")
            print(f"  CSV: {csv_file}")

        return summary_data


def main():
    parser = argparse.ArgumentParser(description='Staged TF-DWT Parameter Optimization')
    parser.add_argument('--focus', choices=['P3_small', 'AVO_small'], required=True,
                       help='Focus case: P3_small or AVO_small')

    args = parser.parse_args()

    optimizer = StagedOptimizer(args.focus)
    optimizer.run_full_optimization()


if __name__ == '__main__':
    main()