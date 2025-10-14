#!/usr/bin/env python3
"""
Script to reproduce TF-DWT experiment results from 20250924_204648

This script automatically:
1. Backs up the current config.py
2. Loads the reproduction configuration
3. Runs the TF-DWT experiment
4. Restores the original configuration

Expected results:
- Overall accuracy: 0.6389 ± 0.0180
- 95% Confidence Interval: [0.6315, 0.6463]
- P3 Dataset - Mean Accuracy: 0.5884 | 95% CI: [0.5743, 0.6024]
- AVO Dataset - Mean Accuracy: 0.6523 | 95% CI: [0.6432, 0.6614]
"""

import os
import sys
import shutil
import subprocess
from datetime import datetime

def main():
    print("=" * 60)
    print("TF-DWT Experiment Reproduction Script")
    print("Target: tfdwt_summary_stats_20250924_204648.csv results")
    print("=" * 60)

    # File paths
    original_config = "config.py"
    reproduction_config = "config_reproduce_tfdwt_20250924_204648.py"
    backup_config = f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"

    # Check if reproduction config exists
    if not os.path.exists(reproduction_config):
        print(f"ERROR: Reproduction config file not found: {reproduction_config}")
        print("Please ensure config_reproduce_tfdwt_20250924_204648.py exists in the current directory.")
        return 1

    # Check if main_tfdwt.py exists
    if not os.path.exists("main_tfdwt.py"):
        print("ERROR: main_tfdwt.py not found in current directory.")
        print("Please run this script from the EEG_experiments directory.")
        return 1

    try:
        # Step 1: Backup original config
        print(f"Step 1: Backing up original config.py as {backup_config}")
        if os.path.exists(original_config):
            shutil.copy2(original_config, backup_config)
            print(f"✓ Original config backed up to {backup_config}")
        else:
            print("WARNING: No existing config.py found to backup")

        # Step 2: Load reproduction config
        print(f"Step 2: Loading reproduction configuration...")
        shutil.copy2(reproduction_config, original_config)
        print(f"✓ Reproduction config loaded as config.py")

        # Step 3: Display expected results
        print("\nStep 3: Expected Results Summary:")
        print("- Overall accuracy: 0.6389 ± 0.0180")
        print("- 95% Confidence Interval: [0.6315, 0.6463]")
        print("- P3 Dataset - Mean Accuracy: 0.5884 | 95% CI: [0.5743, 0.6024]")
        print("- AVO Dataset - Mean Accuracy: 0.6523 | 95% CI: [0.6432, 0.6614]")

        # Step 4: Run experiment
        print(f"\nStep 4: Running TF-DWT experiment...")
        print("This may take several hours depending on your hardware.")
        print("Starting main_tfdwt.py...")
        print("-" * 60)

        # Run the experiment
        result = subprocess.run([sys.executable, "main_tfdwt.py"],
                              capture_output=False,
                              text=True)

        print("-" * 60)
        if result.returncode == 0:
            print("✓ Experiment completed successfully!")
        else:
            print(f"✗ Experiment failed with return code: {result.returncode}")
            return result.returncode

    except Exception as e:
        print(f"ERROR: An exception occurred: {e}")
        return 1

    finally:
        # Step 5: Restore original config
        print(f"\nStep 5: Restoring original configuration...")
        try:
            if os.path.exists(backup_config):
                shutil.copy2(backup_config, original_config)
                print(f"✓ Original config.py restored from {backup_config}")
            else:
                print("WARNING: No backup config found to restore")
        except Exception as e:
            print(f"WARNING: Could not restore original config: {e}")
            print(f"Manual restore required: cp {backup_config} config.py")

    print("\n" + "=" * 60)
    print("Reproduction attempt completed!")
    print("Check the generated CSV files for results comparison.")
    print("=" * 60)

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)