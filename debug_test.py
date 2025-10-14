#!/usr/bin/env python3
"""Debug the TF-DWT output to see what's happening."""

import os
import sys
import subprocess
import re

def debug_test():
    # Set environment for test config
    env = os.environ.copy()
    env['CONFIG_OVERRIDE_PATH'] = 'config.py'  # Use original config

    print("Running main_tfdwt.py with debugging...")
    cmd = [sys.executable, 'main_tfdwt.py']

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800  # 30 min timeout
    )

    print(f"Return code: {result.returncode}")
    print(f"Stdout length: {len(result.stdout)} chars")
    print(f"Stderr length: {len(result.stderr)} chars")

    # Print full stdout
    print("\n=== FULL STDOUT ===")
    print(result.stdout)

    # Print stderr (warnings, etc.)
    if result.stderr:
        print("\n=== STDERR (first 2000 chars) ===")
        print(result.stderr[:2000])

    # Try to find log files
    import glob
    log_files = glob.glob("log_*/*.log")
    print(f"\nFound log files: {log_files}")

    for log_file in log_files[:2]:  # Check first 2 log files
        print(f"\n=== LOG FILE: {log_file} (last 1000 chars) ===")
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                print(content[-1000:])
        except Exception as e:
            print(f"Error reading {log_file}: {e}")

if __name__ == "__main__":
    debug_test()