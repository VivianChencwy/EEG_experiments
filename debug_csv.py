#!/usr/bin/env python3
"""
Debug script to test the save_csv_results function and nested_results structure
"""

import sys
import os
sys.path.insert(0, '/home/vivian/eeg/EEG_experiments')

from main import save_csv_results

# Test data structure similar to what nested_cv should return
test_nested_results = {
    'detailed_fold_results': [
        {
            'repeat': 1,
            'fold': 1,
            'overall_accuracy': 0.75,
            'p3_accuracy': 0.70,
            'avo_accuracy': 0.80,
            'p3_precision': 0.72,
            'p3_recall': 0.68,
            'p3_f1': 0.70,
            'p3_auc': 0.75,
            'avo_precision': 0.82,
            'avo_recall': 0.78,
            'avo_f1': 0.80,
            'avo_auc': 0.85,
            'p3_test_size': 100,
            'avo_test_size': 200
        },
        {
            'repeat': 1,
            'fold': 2,
            'overall_accuracy': 0.73,
            'p3_accuracy': 0.68,
            'avo_accuracy': 0.78,
            'p3_precision': 0.70,
            'p3_recall': 0.66,
            'p3_f1': 0.68,
            'p3_auc': 0.73,
            'avo_precision': 0.80,
            'avo_recall': 0.76,
            'avo_f1': 0.78,
            'avo_auc': 0.83,
            'p3_test_size': 95,
            'avo_test_size': 205
        }
    ],
    'mean_accuracy': 0.74,
    'std_accuracy': 0.01,
    'other_metrics': {
        'precision': {'mean': 0.76, 'std': 0.02},
        'recall': {'mean': 0.72, 'std': 0.02}
    }
}

# Test the save_csv_results function
print("Testing save_csv_results function...")
try:
    save_csv_results(test_nested_results, 'debug_test', None)
    print("✓ save_csv_results function works correctly")
except Exception as e:
    print(f"✗ Error in save_csv_results: {e}")
    import traceback
    traceback.print_exc()

# Check if CSV file was created
import glob
csv_files = glob.glob('/home/vivian/eeg/EEG_experiments/debug_test*detailed_results*.csv')
if csv_files:
    print(f"✓ CSV file created: {csv_files[0]}")
    # Show first few lines
    with open(csv_files[0], 'r') as f:
        lines = f.readlines()
        print("First few lines of CSV:")
        for i, line in enumerate(lines[:3]):
            print(f"  {i+1}: {line.strip()}")
else:
    print("✗ No CSV file was created")
