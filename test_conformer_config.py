#!/usr/bin/env python3
"""
Quick test script for EEGConformer configurations
"""

import subprocess
import sys
import time

def update_conformer_config(embedding_dim=32, num_heads=8, num_layers=3, 
                           conv_spatial_dim=32, conv_temporal_dim=25,
                           activation='gelu', learning_rate=0.0005, dropout=0.3):
    """Update EEGConformer configuration in config.py"""
    
    # Read current config
    with open('config.py', 'r') as f:
        lines = f.readlines()
    
    # Update specific parameters
    updates = {
        'CONFORMER_EMBEDDING_DIM': embedding_dim,
        'CONFORMER_NUM_HEADS': num_heads, 
        'CONFORMER_NUM_LAYERS': num_layers,
        'CONFORMER_CONV_SPATIAL_DIM': conv_spatial_dim,
        'CONFORMER_CONV_TEMPORAL_DIM': conv_temporal_dim,
        'CONFORMER_ACTIVATION': f"'{activation}'",
        'LEARNING_RATE': learning_rate,
        'DROPOUT_RATE': dropout
    }
    
    # Update lines
    for i, line in enumerate(lines):
        for param, value in updates.items():
            if line.strip().startswith(f'{param} ='):
                lines[i] = f'{param} = {value}  # Updated by test script\n'
                break
    
    # Write updated config
    with open('config.py', 'w') as f:
        f.writelines(lines)
    
    print(f"Updated config with: emb_dim={embedding_dim}, heads={num_heads}, "
          f"layers={num_layers}, lr={learning_rate}, dropout={dropout}")

def run_experiment_quick():
    """Run experiment and extract accuracy."""
    try:
        result = subprocess.run([sys.executable, 'main.py'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None
            
        # Extract accuracy
        output = result.stdout
        lines = output.split('\n')
        
        for i, line in enumerate(lines):
            if "Combined Model (All Subjects)" in line:
                try:
                    # Look for accuracy in next few lines
                    for j in range(1, 5):
                        if i+j < len(lines) and "Mean Accuracy:" in lines[i+j]:
                            acc_line = lines[i+j]
                            accuracy = float(acc_line.split("Mean Accuracy:")[1].strip())
                            return accuracy
                except (IndexError, ValueError):
                    continue
                    
        return None
        
    except subprocess.TimeoutExpired:
        print("Experiment timed out")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_configurations():
    """Test several promising configurations."""
    
    configs = [
        # Config 1: Smaller, efficient model
        {'embedding_dim': 32, 'num_heads': 8, 'num_layers': 2, 
         'conv_spatial_dim': 32, 'learning_rate': 0.001, 'dropout': 0.2},
        
        # Config 2: Balanced model 
        {'embedding_dim': 40, 'num_heads': 8, 'num_layers': 3, 
         'conv_spatial_dim': 40, 'learning_rate': 0.0005, 'dropout': 0.3},
         
        # Config 3: Larger model
        {'embedding_dim': 64, 'num_heads': 8, 'num_layers': 4, 
         'conv_spatial_dim': 56, 'learning_rate': 0.0003, 'dropout': 0.4},
         
        # Config 4: More attention heads
        {'embedding_dim': 48, 'num_heads': 12, 'num_layers': 3, 
         'conv_spatial_dim': 40, 'learning_rate': 0.0005, 'dropout': 0.3}
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Testing Configuration {i+1}/4")
        print(f"Config: {config}")
        print(f"{'='*60}")
        
        # Update config
        update_conformer_config(**config)
        
        # Run experiment
        start_time = time.time()
        accuracy = run_experiment_quick()
        duration = time.time() - start_time
        
        if accuracy is not None:
            print(f"✅ Accuracy: {accuracy:.4f} (Duration: {duration:.1f}s)")
            results.append((config, accuracy))
        else:
            print(f"❌ Failed to get accuracy")
            
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        for i, (config, acc) in enumerate(results):
            print(f"{i+1}. Accuracy: {acc:.4f} - {config}")
            
        # Update with best config
        best_config, best_acc = results[0]
        print(f"\nBest configuration (accuracy: {best_acc:.4f}):")
        print(best_config)
        
        update_conformer_config(**best_config)
        print("\nConfig updated with best parameters!")
    else:
        print("No successful runs")

if __name__ == "__main__":
    print("Testing EEGConformer configurations...")
    test_configurations()