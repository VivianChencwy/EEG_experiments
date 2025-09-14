#!/usr/bin/env python3
"""
EEGConformer parameter optimization script
"""

import os
import json
import time
import itertools
from datetime import datetime

# Parameter grid for optimization
PARAM_GRID = {
    'embedding_dim': [24, 32, 40, 56, 64],
    'num_heads': [4, 8, 10, 12],  # Must divide embedding_dim evenly
    'num_layers': [2, 3, 4, 5],
    'conv_spatial_dim': [24, 32, 40, 56],
    'conv_temporal_dim': [16, 25, 32],
    'activation': ['relu', 'gelu', 'swish'],
    'learning_rate': [0.0003, 0.0005, 0.001],
    'dropout': [0.2, 0.3, 0.4, 0.5]
}

def is_valid_config(config):
    """Check if the configuration is valid (num_heads must divide embedding_dim)."""
    return config['embedding_dim'] % config['num_heads'] == 0

def update_config_file(config):
    """Update the config.py file with new parameters."""
    
    # Read current config
    with open('config.py', 'r') as f:
        lines = f.readlines()
    
    # Update specific parameters
    updates = {
        'CONFORMER_EMBEDDING_DIM': config['embedding_dim'],
        'CONFORMER_NUM_HEADS': config['num_heads'], 
        'CONFORMER_NUM_LAYERS': config['num_layers'],
        'CONFORMER_CONV_SPATIAL_DIM': config['conv_spatial_dim'],
        'CONFORMER_CONV_TEMPORAL_DIM': config['conv_temporal_dim'],
        'CONFORMER_ACTIVATION': f"'{config['activation']}'",
        'LEARNING_RATE': config['learning_rate'],
        'DROPOUT_RATE': config['dropout']
    }
    
    # Update lines
    for i, line in enumerate(lines):
        for param, value in updates.items():
            if line.strip().startswith(f'{param} ='):
                lines[i] = f'{param} = {value}  # Auto-updated by optimizer\n'
                break
    
    # Write updated config
    with open('config.py', 'w') as f:
        f.writelines(lines)

def run_experiment():
    """Run the main experiment and return results."""
    import subprocess
    import sys
    
    try:
        # Run the experiment with timeout
        result = subprocess.run([sys.executable, 'main.py'], 
                              capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"Experiment failed with error: {result.stderr}")
            return None
            
        # Parse accuracy from output
        output = result.stdout
        combined_acc = None
        
        # Look for "Combined Model (All Subjects)" accuracy
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if "Combined Model (All Subjects)" in line and "Mean Accuracy:" in lines[i+1]:
                acc_line = lines[i+1]
                try:
                    combined_acc = float(acc_line.split("Mean Accuracy:")[1].strip())
                    break
                except (IndexError, ValueError):
                    continue
                    
        return combined_acc
        
    except subprocess.TimeoutExpired:
        print("Experiment timed out")
        return None
    except Exception as e:
        print(f"Error running experiment: {e}")
        return None

def optimize_parameters(max_trials=50, results_file='conformer_optimization_results.json'):
    """Optimize EEGConformer parameters using random search."""
    
    results = []
    best_accuracy = 0.0
    best_config = None
    
    # Generate valid configurations
    configs = []
    param_combinations = list(itertools.product(*PARAM_GRID.values()))
    
    # Create configuration dictionaries and filter valid ones
    for combo in param_combinations:
        config = dict(zip(PARAM_GRID.keys(), combo))
        if is_valid_config(config):
            configs.append(config)
    
    print(f"Generated {len(configs)} valid configurations")
    
    # Randomly sample configurations to test
    import random
    random.shuffle(configs)
    configs = configs[:max_trials]
    
    print(f"Testing {len(configs)} configurations")
    
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Trial {i+1}/{len(configs)}")
        print(f"Config: {config}")
        print(f"{'='*60}")
        
        # Update config file
        update_config_file(config)
        
        # Run experiment
        start_time = time.time()
        accuracy = run_experiment()
        duration = time.time() - start_time
        
        if accuracy is not None:
            print(f"Accuracy: {accuracy:.4f} (Duration: {duration:.1f}s)")
            
            # Record results
            result = {
                'trial': i + 1,
                'config': config,
                'accuracy': accuracy,
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
            
            # Update best configuration
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = config
                print(f"NEW BEST: {accuracy:.4f}")
        else:
            print("Failed to get accuracy")
        
        # Save intermediate results
        with open(results_file, 'w') as f:
            json.dump({
                'results': results,
                'best_config': best_config,
                'best_accuracy': best_accuracy,
                'total_trials': len(configs)
            }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Best config: {best_config}")
    
    # Update config with best parameters
    if best_config:
        update_config_file(best_config)
        print("Config file updated with best parameters")
    
    return results, best_config, best_accuracy

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize EEGConformer parameters')
    parser.add_argument('--max_trials', type=int, default=20,
                       help='Maximum number of trials to run')
    parser.add_argument('--results_file', type=str, 
                       default=f'conformer_optimization_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    print(f"Starting EEGConformer parameter optimization")
    print(f"Max trials: {args.max_trials}")
    print(f"Results file: {args.results_file}")
    
    results, best_config, best_accuracy = optimize_parameters(
        max_trials=args.max_trials,
        results_file=args.results_file
    )
    
    print(f"\nOptimization completed. Results saved to {args.results_file}")