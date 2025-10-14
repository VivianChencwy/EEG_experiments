#!/usr/bin/env python3
"""
Test a single hyperparameter configuration to verify the system works.
This runs a simplified version without nested CV for debugging.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

def create_test_config():
    """Create a test configuration with good parameters."""
    test_params = {
        'LEARNING_RATE': 0.01,
        'WEIGHT_DECAY': 1e-4,
        'DROPOUT_RATE': 0.25,
        'BATCH_SIZE': 32,
        'MAX_EPOCHS': 50,  # Reduced for testing
        'EARLY_STOPPING_PATIENCE': 10,  # Reduced for testing
        'NOISE_STD': 0.005,
        'TIME_SHIFT_RANGE': 5,
        'LABEL_SMOOTHING': 0.05,
        'classifier': 'EEGConformer',
    }

    # Read base config
    with open('config.py', 'r') as f:
        config_content = f.read()

    # Create test config
    temp_config_path = 'test_config.py'
    modified_content = config_content

    # Replace parameter values
    import re
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
        if param in test_params:
            pattern = rf'^{param}\s*=.*$'
            replacement = template.format(test_params[param])
            modified_content = re.sub(pattern, replacement, modified_content, flags=re.MULTILINE)

    # Handle classifier
    pattern = r"^classifier\s*=.*$"
    replacement = f"classifier = '{test_params['classifier']}'"
    modified_content = re.sub(pattern, replacement, modified_content, flags=re.MULTILINE)

    # Ensure trial configuration
    if 'NESTED_CV_TRIALS_PER_SUBJECT_P3' not in modified_content:
        modified_content += "\nNESTED_CV_TRIALS_PER_SUBJECT_P3 = 20\n"
    if 'NESTED_CV_TRIALS_PER_SUBJECT_AVO' not in modified_content:
        modified_content += "\nNESTED_CV_TRIALS_PER_SUBJECT_AVO = 200\n"

    with open(temp_config_path, 'w') as f:
        f.write(modified_content)

    return temp_config_path, test_params


def run_test():
    """Run a single test trial."""
    print("Creating test configuration...")
    test_config_path, test_params = create_test_config()

    print(f"Test parameters: {test_params}")

    try:
        # Set environment for config override
        env = os.environ.copy()
        env['CONFIG_OVERRIDE_PATH'] = test_config_path

        print("Running TF-DWT with test configuration...")
        cmd = [sys.executable, 'main_tfdwt.py']

        print(f"Command: {' '.join(cmd)}")
        print("This may take 10-30 minutes for the nested CV...")

        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        print(f"Process completed with return code: {result.returncode}")
        print(f"Stdout length: {len(result.stdout)} chars")
        print(f"Stderr length: {len(result.stderr)} chars")

        if result.returncode != 0:
            print("STDERR:")
            print(result.stderr)
            return False
        else:
            print("STDOUT (last 1000 chars):")
            print(result.stdout[-1000:])

            # Try to extract accuracy
            import re
            patterns = [
                r'Overall accuracy:\s+([0-9.]+)',
                r'mean_accuracy.*?([0-9.]+)',
                r'Mean Accuracy:\s+([0-9.]+)',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, result.stdout, re.IGNORECASE)
                if matches:
                    accuracy = float(matches[-1])
                    print(f"✓ Found accuracy: {accuracy:.4f}")
                    return True

            print("⚠ Process completed but could not extract accuracy")
            return False

    except subprocess.TimeoutExpired:
        print("✗ Process timed out after 1 hour")
        return False
    except Exception as e:
        print(f"✗ Process failed with exception: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(test_config_path):
            os.unlink(test_config_path)


if __name__ == "__main__":
    print("=== Single Trial Test for TF-DWT ===")
    success = run_test()
    if success:
        print("\n✓ Test completed successfully! The hyperparameter tuning system should work.")
    else:
        print("\n✗ Test failed. Please check the configuration and fix issues before running full tuning.")