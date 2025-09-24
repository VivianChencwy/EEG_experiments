#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for TF-DWT (Target-Focused Domain Weighted Training)

This script performs hyperparameter optimization for main_tfdwt.py to maximize
accuracy under the configuration:
- NESTED_CV_TRIALS_PER_SUBJECT_P3 = 20
- NESTED_CV_TRIALS_PER_SUBJECT_AVO = 200

Supports multiple search strategies:
- Random Search: Fast exploration of parameter space
- Grid Search: Exhaustive search over specified parameter grid
- Bayesian Optimization: Smart search using Gaussian Process (requires scikit-optimize)

Run: python tune_tfdwt.py --strategy random --n_trials 50
"""

import os
import sys
import json
import argparse
import time
import shutil
import tempfile
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Import for parameter sampling
try:
    from sklearn.model_selection import ParameterSampler
    from scipy.stats import uniform, randint, loguniform
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try to import Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False


class TFDWTTuner:
    """Hyperparameter tuner for TF-DWT method."""

    def __init__(self, base_config_path: str = "config.py", results_dir: str = "tuning_results"):
        self.base_config_path = Path(base_config_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Setup logging
        log_path = self.results_dir / f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Results tracking
        self.results_file = self.results_dir / "tuning_results.json"
        self.best_params = None
        self.best_score = -float('inf')
        self.trial_results = []

    def get_parameter_space(self) -> Dict[str, Any]:
        """Define the hyperparameter search space."""
        return {
            # Core training parameters
            'LEARNING_RATE': {
                'type': 'log_uniform',
                'low': 1e-4,
                'high': 1e-1,
                'default': 0.01
            },
            'WEIGHT_DECAY': {
                'type': 'log_uniform',
                'low': 1e-6,
                'high': 1e-2,
                'default': 1e-4
            },
            'DROPOUT_RATE': {
                'type': 'uniform',
                'low': 0.1,
                'high': 0.5,
                'default': 0.25
            },
            'BATCH_SIZE': {
                'type': 'choice',
                'values': [16, 24, 32, 48, 64],
                'default': 32
            },

            # Training schedule
            'MAX_EPOCHS': {
                'type': 'int_uniform',
                'low': 200,
                'high': 800,
                'default': 500
            },
            'EARLY_STOPPING_PATIENCE': {
                'type': 'int_uniform',
                'low': 20,
                'high': 100,
                'default': 50
            },

            # Data augmentation
            'NOISE_STD': {
                'type': 'uniform',
                'low': 0.001,
                'high': 0.02,
                'default': 0.005
            },
            'TIME_SHIFT_RANGE': {
                'type': 'int_uniform',
                'low': 2,
                'high': 15,
                'default': 5
            },
            'LABEL_SMOOTHING': {
                'type': 'uniform',
                'low': 0.0,
                'high': 0.2,
                'default': 0.05
            },

            # Model architecture
            'classifier': {
                'type': 'choice',
                'values': ['EEGConformer', 'EEGNetv4', 'ShallowFBCSPNet', 'SepConv1DLite', 'EEGChannelNet'],
                'default': 'EEGConformer'
            },

            # TF-DWT specific parameters (will be embedded in main_tfdwt.py modifications)
            'w_small_clip_max': {
                'type': 'uniform',
                'low': 2.0,
                'high': 8.0,
                'default': 6.0
            },
            'lambda_mmd_base': {
                'type': 'uniform',
                'low': 0.05,
                'high': 0.5,
                'default': 0.1
            },
            'gradient_clip_norm': {
                'type': 'uniform',
                'low': 1.0,
                'high': 10.0,
                'default': 5.0
            },
            'warmup_ratio': {
                'type': 'uniform',
                'low': 0.05,
                'high': 0.25,
                'default': 0.1
            }
        }

    def sample_parameters(self, strategy: str = 'random', n_trials: int = 50) -> List[Dict[str, Any]]:
        """Sample parameters based on chosen strategy."""
        param_space = self.get_parameter_space()

        if strategy == 'random':
            return self._random_sample(param_space, n_trials)
        elif strategy == 'grid':
            return self._grid_sample(param_space)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _random_sample(self, param_space: Dict[str, Any], n_trials: int) -> List[Dict[str, Any]]:
        """Generate random parameter samples."""
        samples = []

        for _ in range(n_trials):
            sample = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'uniform':
                    sample[param_name] = np.random.uniform(param_config['low'], param_config['high'])
                elif param_config['type'] == 'log_uniform':
                    sample[param_name] = np.exp(np.random.uniform(
                        np.log(param_config['low']), np.log(param_config['high'])
                    ))
                elif param_config['type'] == 'int_uniform':
                    sample[param_name] = np.random.randint(param_config['low'], param_config['high'] + 1)
                elif param_config['type'] == 'choice':
                    sample[param_name] = np.random.choice(param_config['values'])
                else:
                    sample[param_name] = param_config['default']
            samples.append(sample)

        return samples

    def _grid_sample(self, param_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate grid search parameter combinations (reduced for feasibility)."""
        # Simplified grid for faster execution
        grid = {
            'LEARNING_RATE': [0.001, 0.01, 0.05],
            'WEIGHT_DECAY': [1e-5, 1e-4, 1e-3],
            'DROPOUT_RATE': [0.15, 0.25, 0.35],
            'BATCH_SIZE': [24, 32, 48],
            'classifier': ['EEGConformer', 'SepConv1DLite'],
            'w_small_clip_max': [3.0, 6.0],
            'lambda_mmd_base': [0.1, 0.2],
        }

        # Generate all combinations
        import itertools
        keys = list(grid.keys())
        values = list(grid.values())
        combinations = list(itertools.product(*values))

        samples = []
        for combo in combinations:
            sample = dict(zip(keys, combo))
            # Fill in defaults for parameters not in grid
            for param_name, param_config in param_space.items():
                if param_name not in sample:
                    sample[param_name] = param_config['default']
            samples.append(sample)

        return samples

    def create_config_for_trial(self, params: Dict[str, Any], trial_id: int) -> str:
        """Create a temporary config file with specified parameters."""
        # Read base config
        with open(self.base_config_path, 'r') as f:
            config_content = f.read()

        # Create temporary config file
        temp_config_path = self.results_dir / f"config_trial_{trial_id}.py"

        # Replace parameter values in config
        modified_content = config_content

        # Update basic parameters
        config_mappings = {
            'LEARNING_RATE': 'LEARNING_RATE = {:.6f}',
            'WEIGHT_DECAY': 'WEIGHT_DECAY = {:.2e}',
            'DROPOUT_RATE': 'DROPOUT_RATE = {:.3f}',
            'BATCH_SIZE': 'BATCH_SIZE = {}',
            'MAX_EPOCHS': 'MAX_EPOCHS = {}',
            'EARLY_STOPPING_PATIENCE': 'EARLY_STOPPING_PATIENCE = {}',
            'NOISE_STD': 'NOISE_STD = {:.4f}',
            'TIME_SHIFT_RANGE': 'TIME_SHIFT_RANGE = {}',
            'LABEL_SMOOTHING': 'LABEL_SMOOTHING = {:.3f}',
        }

        for param, template in config_mappings.items():
            if param in params:
                # Find and replace the line
                import re
                pattern = rf'^{param}\s*=.*$'
                replacement = template.format(params[param])
                modified_content = re.sub(pattern, replacement, modified_content, flags=re.MULTILINE)

        # Handle classifier specially
        if 'classifier' in params:
            pattern = r"^classifier\s*=.*$"
            replacement = f"classifier = '{params['classifier']}'"
            modified_content = re.sub(pattern, replacement, modified_content, flags=re.MULTILINE)

        # Ensure we keep the required trial configuration
        if 'NESTED_CV_TRIALS_PER_SUBJECT_P3' not in modified_content:
            modified_content += "\nNESTED_CV_TRIALS_PER_SUBJECT_P3 = 20\n"
        if 'NESTED_CV_TRIALS_PER_SUBJECT_AVO' not in modified_content:
            modified_content += "\nNESTED_CV_TRIALS_PER_SUBJECT_AVO = 200\n"

        # Write temporary config
        with open(temp_config_path, 'w') as f:
            f.write(modified_content)

        return str(temp_config_path)

    def create_modified_main_tfdwt(self, params: Dict[str, Any], trial_id: int) -> str:
        """Create a modified version of main_tfdwt.py with TF-DWT specific parameters."""
        # Read original main_tfdwt.py
        with open('main_tfdwt.py', 'r') as f:
            main_content = f.read()

        # Create temporary modified version
        temp_main_path = self.results_dir / f"main_tfdwt_trial_{trial_id}.py"

        # Replace TF-DWT specific parameters
        modified_content = main_content

        # Add sys.path fix after the docstring but before imports
        import_fix = """
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""

        # Find the end of the initial docstring and insert the fix
        docstring_end = main_content.find('"""', main_content.find('"""') + 3) + 3
        modified_content = main_content[:docstring_end] + import_fix + main_content[docstring_end:]

        # Modify w_small clipping (line 204, 318)
        if 'w_small_clip_max' in params:
            import re
            # Replace the clipping value in line around 204 and 318
            pattern = r'w_small_target = min\(w_small_target, \d+\.?\d*\)'
            replacement = f'w_small_target = min(w_small_target, {params["w_small_clip_max"]:.1f})'
            modified_content = re.sub(pattern, replacement, modified_content)

        # Modify lambda_mmd base values (lines 207)
        if 'lambda_mmd_base' in params:
            pattern = r'lambda_mmd = 0\.1 if overall_ratio < 2\.0 else \(0\.2 if overall_ratio < 4\.0 else 0\.3\)'
            base = params['lambda_mmd_base']
            replacement = f'lambda_mmd = {base:.3f} if overall_ratio < 2.0 else ({base*2:.3f} if overall_ratio < 4.0 else {base*3:.3f})'
            modified_content = re.sub(pattern, replacement, modified_content)

        # Modify gradient clipping norm (line 402)
        if 'gradient_clip_norm' in params:
            pattern = r'torch\.nn\.utils\.clip_grad_norm_\(model\.parameters\(\), max_norm=\d+\.?\d*\)'
            replacement = f'torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm={params["gradient_clip_norm"]:.1f})'
            modified_content = re.sub(pattern, replacement, modified_content)

        # Modify warmup ratio (line 208)
        if 'warmup_ratio' in params:
            pattern = r'warmup = max\(2, min\(5, int\(0\.1 \* MAX_EPOCHS\)\)\)'
            replacement = f'warmup = max(2, min(10, int({params["warmup_ratio"]:.3f} * MAX_EPOCHS)))'
            modified_content = re.sub(pattern, replacement, modified_content)

        # Write temporary modified file
        with open(temp_main_path, 'w') as f:
            f.write(modified_content)

        return str(temp_main_path)

    def convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def run_trial(self, params: Dict[str, Any], trial_id: int) -> Tuple[float, Dict[str, Any]]:
        """Run a single hyperparameter trial."""
        # Convert numpy types for JSON serialization
        serializable_params = self.convert_numpy_types(params)

        # Log trial start with better formatting
        print(f"\n{'='*80}")
        print(f"TRIAL {trial_id + 1} STARTING")
        print(f"{'='*80}")
        print(f"Key Parameters:")
        key_params = ['LEARNING_RATE', 'classifier', 'w_small_clip_max', 'lambda_mmd_base', 'BATCH_SIZE']
        for key in key_params:
            if key in params:
                if isinstance(params[key], float):
                    print(f"   {key}: {params[key]:.4f}")
                else:
                    print(f"   {key}: {params[key]}")

        self.logger.info(f"Starting trial {trial_id} with params: {json.dumps(serializable_params, indent=2)}")

        import time
        trial_start_time = time.time()
        print(f"Trial started at: {time.strftime('%H:%M:%S')}")
        print(f"Expected duration: 15-30 minutes per trial")
        print(f"Monitor logs: tail -f log_0909/TF_DWT_*.log")
        print("="*80)

        try:
            # Create temporary config and main files
            temp_config_path = self.create_config_for_trial(params, trial_id)
            temp_main_path = self.create_modified_main_tfdwt(params, trial_id)

            # Set environment variable for config override
            env = os.environ.copy()
            env['CONFIG_OVERRIDE_PATH'] = temp_config_path

            # Run the experiment
            import subprocess
            cmd = [sys.executable, temp_main_path]

            start_time = time.time()
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=10800  # 3 hour timeout for nested CV
            )
            duration = time.time() - start_time

            trial_end_time = time.time()
            trial_duration_min = (trial_end_time - trial_start_time) / 60

            if result.returncode != 0:
                print(f"\nTRIAL {trial_id + 1} FAILED after {trial_duration_min:.1f} minutes")
                print(f"Error: {result.stderr[:200]}...")
                self.logger.error(f"Trial {trial_id} failed with error: {result.stderr}")
                return -1.0, {'error': result.stderr, 'duration': duration}

            # Parse results from stdout/log files
            accuracy = self._extract_accuracy_from_output(result.stdout, trial_id)

            # Print trial completion
            print(f"\n{'='*80}")
            print(f"TRIAL {trial_id + 1} COMPLETED")
            print(f"{'='*80}")
            print(f"Duration: {trial_duration_min:.1f} minutes")
            print(f"Accuracy: {accuracy:.4f}")
            if accuracy > 0:
                print(f"Valid result obtained!")
            else:
                print(f"Could not extract accuracy from output")
            print("="*80)

            self.logger.info(f"Trial {trial_id} completed with accuracy: {accuracy:.4f}")

            # Clean up temporary files
            os.unlink(temp_config_path)
            os.unlink(temp_main_path)

            return accuracy, {'duration': duration, 'stdout': result.stdout[:1000]}  # Truncate output

        except subprocess.TimeoutExpired:
            self.logger.error(f"Trial {trial_id} timed out")
            return -1.0, {'error': 'timeout'}
        except Exception as e:
            self.logger.error(f"Trial {trial_id} failed with exception: {e}")
            return -1.0, {'error': str(e)}

    def _extract_accuracy_from_output(self, output: str, trial_id: int) -> float:
        """Extract accuracy from the experiment output."""
        import re

        # Enhanced patterns to catch different output formats
        patterns = [
            r'Overall accuracy:\s+([0-9.]+)',
            r'mean_accuracy[\'\"]*\s*[:\s=]+\s*([0-9.]+)',
            r'Mean Accuracy:\s+([0-9.]+)',
            r'Best accuracy:\s+([0-9.]+)',
            r'Test.*Acc[=:]\s*([0-9.]+)',
            r'accuracy[\'\"]*\s*[:\s=]+\s*([0-9.]+)',
            r'Accuracy[\'\"]*\s*[:\s=]+\s*([0-9.]+)',
            r'final.*accuracy[\'\"]*\s*[:\s=]+\s*([0-9.]+)',
            r'validation.*accuracy[\'\"]*\s*[:\s=]+\s*([0-9.]+)',
        ]

        # Check stdout first
        for pattern in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                # Take the last/highest accuracy found
                accuracies = [float(match) for match in matches if float(match) <= 1.0]  # Filter realistic accuracies
                if accuracies:
                    return max(accuracies)

        # Try to find any log files in the results directory AND log_0909/ that might contain accuracy
        log_dirs = [
            self.results_dir,
            Path("."),
            Path("log_0909"),
            Path("log_batch"),
        ]
        log_files = []
        for log_dir in log_dirs:
            if log_dir.exists():
                log_files.extend(list(log_dir.glob("*.log")))
                log_files.extend(list(log_dir.glob("**/TF_DWT*.log")))

        # Sort by modification time to get the most recent first
        log_files = sorted([f for f in log_files if f.exists()], key=lambda x: x.stat().st_mtime, reverse=True)

        for log_file in log_files[:10]:  # Check top 10 most recent log files
            try:
                with open(log_file, 'r') as f:
                    log_content = f.read()
                for pattern in patterns:
                    matches = re.findall(pattern, log_content, re.IGNORECASE)
                    if matches:
                        accuracies = [float(match) for match in matches if float(match) <= 1.0]
                        if accuracies:
                            self.logger.info(f"Found accuracy {max(accuracies):.4f} in log file: {log_file}")
                            return max(accuracies)
            except Exception:
                continue

        # If still no accuracy found, check if the process completed successfully
        # but the output was truncated - try to infer from any numerical values
        number_pattern = r'([0-9.]+)'
        numbers = re.findall(number_pattern, output)
        potential_accuracies = []
        for num_str in numbers[-10:]:  # Check last 10 numbers
            try:
                num = float(num_str)
                if 0.0 <= num <= 1.0 and len(num_str) >= 4:  # Likely an accuracy
                    potential_accuracies.append(num)
            except:
                continue

        if potential_accuracies:
            self.logger.info(f"Trial {trial_id}: Inferred accuracy from numbers: {max(potential_accuracies):.4f}")
            return max(potential_accuracies)

        self.logger.warning(f"Could not extract accuracy from trial {trial_id} output (length: {len(output)} chars)")
        # Print first 200 chars for debugging
        if len(output) > 0:
            self.logger.debug(f"Output sample: {output[:200]}")
        return 0.0

    def run_tuning(self, strategy: str = 'random', n_trials: int = 50) -> Dict[str, Any]:
        """Run hyperparameter tuning with specified strategy."""
        self.logger.info(f"Starting hyperparameter tuning with {strategy} strategy, {n_trials} trials")

        # Load existing results if available
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                existing_results = json.load(f)
                self.trial_results = existing_results.get('trials', [])
                if existing_results.get('best_params'):
                    self.best_params = existing_results['best_params']
                    self.best_score = existing_results.get('best_score', -float('inf'))

        # Generate parameter samples
        param_samples = self.sample_parameters(strategy, n_trials)

        start_trial_id = len(self.trial_results)

        for i, params in enumerate(param_samples):
            trial_id = start_trial_id + i

            # Run trial
            score, metadata = self.run_trial(params, trial_id)

            # Update results
            trial_result = {
                'trial_id': trial_id,
                'params': self.convert_numpy_types(params),
                'score': float(score) if score != -1.0 else -1.0,
                'metadata': self.convert_numpy_types(metadata),
                'timestamp': datetime.now().isoformat()
            }

            self.trial_results.append(trial_result)

            # Update best parameters
            if score > self.best_score:
                self.best_score = score
                self.best_params = self.convert_numpy_types(params.copy())
                print(f"\nNEW BEST SCORE: {score:.4f}")
                print(f"Previous best: {self.best_score:.4f} -> New best: {score:.4f}")
                self.logger.info(f"New best score: {score:.4f} with params: {json.dumps(self.best_params, indent=2)}")

            # Print overall progress
            completed_trials = len(self.trial_results)
            valid_scores = [t['score'] for t in self.trial_results if t['score'] > 0]
            print(f"\nOVERALL PROGRESS:")
            print(f"   Completed trials: {completed_trials}/{n_trials}")
            print(f"   Valid results: {len(valid_scores)}/{completed_trials}")
            if valid_scores:
                print(f"   Best so far: {max(valid_scores):.4f}")
                print(f"   Average: {sum(valid_scores)/len(valid_scores):.4f}")
            remaining = n_trials - completed_trials
            if remaining > 0:
                avg_time = sum(t['metadata'].get('duration', 0) for t in self.trial_results) / len(self.trial_results) if self.trial_results else 1500
                est_remaining_hours = (remaining * avg_time) / 3600
                print(f"   Estimated remaining time: {est_remaining_hours:.1f} hours")

            # Save intermediate results
            self._save_results()

            # Early stopping if perfect score
            if score >= 0.99:
                print(f"Near-perfect score achieved ({score:.4f}), stopping early!")
                self.logger.info(f"Near-perfect score achieved ({score:.4f}), stopping early")
                break

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(self.trial_results),
            'all_trials': self.trial_results
        }

    def _save_results(self):
        """Save current results to file."""
        results = {
            'best_params': self.convert_numpy_types(self.best_params) if self.best_params else None,
            'best_score': float(self.best_score) if self.best_score != -float('inf') else -1.0,
            'n_trials': len(self.trial_results),
            'trials': self.trial_results,  # Already converted in run_tuning
            'timestamp': datetime.now().isoformat()
        }

        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Also save best parameters as a separate config file
        if self.best_params:
            best_config_path = self.results_dir / "best_config.py"
            self.create_config_for_trial(self.best_params, 'best')
            shutil.copy(self.results_dir / "config_trial_best.py", best_config_path)
            os.unlink(self.results_dir / "config_trial_best.py")

    def generate_report(self) -> str:
        """Generate a summary report of tuning results."""
        if not self.trial_results:
            return "No tuning results available."

        # Create summary statistics
        scores = [t['score'] for t in self.trial_results if t['score'] > 0]
        if not scores:
            return "No successful trials found."

        df = pd.DataFrame(self.trial_results)
        valid_df = df[df['score'] > 0]

        report = f"""
# Hyperparameter Tuning Report for TF-DWT

## Summary
- Total trials: {len(self.trial_results)}
- Successful trials: {len(valid_df)}
- Best accuracy: {self.best_score:.4f}
- Mean accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}

## Best Parameters
```json
{json.dumps(self.best_params, indent=2)}
```

## Top 5 Parameter Combinations
"""

        # Show top 5 results
        top_results = valid_df.nlargest(5, 'score')
        for i, (_, row) in enumerate(top_results.iterrows()):
            report += f"\n### Rank {i+1}: Accuracy = {row['score']:.4f}\n"
            report += f"```json\n{json.dumps(row['params'], indent=2)}\n```\n"

        # Parameter importance analysis (simple correlation)
        report += "\n## Parameter Importance (Correlation with Accuracy)\n"
        numeric_params = {}
        for trial in self.trial_results:
            if trial['score'] > 0:
                for param, value in trial['params'].items():
                    if isinstance(value, (int, float)):
                        if param not in numeric_params:
                            numeric_params[param] = {'values': [], 'scores': []}
                        numeric_params[param]['values'].append(value)
                        numeric_params[param]['scores'].append(trial['score'])

        correlations = []
        for param, data in numeric_params.items():
            if len(data['values']) > 2:
                corr = np.corrcoef(data['values'], data['scores'])[0, 1]
                if not np.isnan(corr):
                    correlations.append((param, corr))

        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        for param, corr in correlations[:10]:
            report += f"- {param}: {corr:.3f}\n"

        return report


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for TF-DWT')
    parser.add_argument('--strategy', choices=['random', 'grid'], default='random',
                        help='Search strategy to use')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of trials for random search')
    parser.add_argument('--results_dir', default='tuning_results',
                        help='Directory to store results')
    parser.add_argument('--config', default='config.py',
                        help='Base config file path')

    args = parser.parse_args()

    # Initialize tuner
    tuner = TFDWTTuner(args.config, args.results_dir)

    # Run tuning
    results = tuner.run_tuning(args.strategy, args.n_trials)

    # Generate and save report
    report = tuner.generate_report()
    report_path = Path(args.results_dir) / "tuning_report.md"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\n{'='*50}")
    print("HYPERPARAMETER TUNING COMPLETED")
    print(f"{'='*50}")
    print(f"Best accuracy: {results['best_score']:.4f}")
    print(f"Best parameters saved to: {args.results_dir}/best_config.py")
    print(f"Full report saved to: {report_path}")
    print(f"Results saved to: {args.results_dir}/tuning_results.json")

    return results


if __name__ == "__main__":
    main()