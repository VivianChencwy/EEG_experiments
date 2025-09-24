#!/usr/bin/env python3
"""
Ultra-fast test with minimal CV for quick validation (5-10 minutes)
"""

import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path

def create_fast_config():
    """Create a config for ultra-fast testing"""
    # Read base config
    with open('config.py', 'r') as f:
        config_content = f.read()

    # Create fast config
    fast_params = """
# Ultra-fast testing parameters
MAX_EPOCHS = 30                    # Reduced from 500
EARLY_STOPPING_PATIENCE = 5        # Reduced from 50
NESTED_CV_OUTER_FOLDS = 3          # Reduced from 5
NESTED_CV_REPEATS = 2              # Reduced from 5
NESTED_CV_TRIALS_PER_SUBJECT_P3 = 10   # Reduced from 20
NESTED_CV_TRIALS_PER_SUBJECT_AVO = 50   # Reduced from 200
BATCH_SIZE = 64                    # Increased for speed
"""

    # Remove existing parameters
    import re
    lines_to_replace = [
        'MAX_EPOCHS', 'EARLY_STOPPING_PATIENCE', 'NESTED_CV_OUTER_FOLDS',
        'NESTED_CV_REPEATS', 'NESTED_CV_TRIALS_PER_SUBJECT_P3', 'NESTED_CV_TRIALS_PER_SUBJECT_AVO'
    ]

    modified_content = config_content
    for param in lines_to_replace:
        pattern = rf'^{param}\s*=.*$'
        modified_content = re.sub(pattern, f'# {param} set below', modified_content, flags=re.MULTILINE)

    # Add fast parameters
    modified_content += fast_params

    # Write fast config
    fast_config_path = 'ultrafast_config.py'
    with open(fast_config_path, 'w') as f:
        f.write(modified_content)

    return fast_config_path

def run_ultrafast_test():
    """Run ultra-fast test"""
    print("⚡ Ultra-Fast TF-DWT Test")
    print("⏱️  Estimated time: 5-10 minutes")
    print("🔧 Using: 30 epochs, 3 folds, 2 repeats, small trial counts")
    print("=" * 50)

    # Create fast config
    fast_config = create_fast_config()

    try:
        # Set environment for fast config
        env = os.environ.copy()
        env['CONFIG_OVERRIDE_PATH'] = os.path.abspath(fast_config)

        cmd = [sys.executable, 'main_tfdwt.py']
        print(f"🚀 Running: {' '.join(cmd)}")

        start_time = time.time()

        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Show progress
        last_progress = time.time()
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                now = time.time()
                elapsed = now - start_time
                print(f"[{elapsed/60:.1f}min] {output.strip()}")
                last_progress = now

        return_code = process.poll()
        stderr = process.stderr.read()
        total_time = time.time() - start_time

        print("=" * 50)
        print(f"🏁 Completed in {total_time/60:.1f} minutes")

        if return_code == 0:
            print("✅ Ultra-fast test successful!")

            # Try to extract accuracy from latest log
            log_files = list(Path("log_0909").glob("TF_DWT*.log")) if Path("log_0909").exists() else []
            if log_files:
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                print(f"📊 Check results in: {latest_log}")

                # Try to extract accuracy
                try:
                    with open(latest_log, 'r') as f:
                        content = f.read()
                    import re
                    matches = re.findall(r'Overall accuracy:\s+([0-9.]+)', content)
                    if matches:
                        accuracy = float(matches[-1])
                        print(f"🎯 Found accuracy: {accuracy:.4f}")
                        return True
                except Exception as e:
                    print(f"⚠️ Could not extract accuracy: {e}")

            print("✅ Test completed - check log files for results")
            return True
        else:
            print("❌ Test failed!")
            if stderr:
                print("Error:")
                print(stderr[-1000:])
            return False

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted")
        process.terminate()
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(fast_config):
            os.unlink(fast_config)

if __name__ == "__main__":
    success = run_ultrafast_test()
    if success:
        print("\n🎉 System is working! You can now run full tuning:")
        print("   python run_quick_with_progress.py     # 5 trials, ~2 hours")
        print("   python run_tuning_example.py --mode standard  # 50 trials, ~8 hours")
    else:
        print("\n❌ Please fix issues before running full tuning")
    sys.exit(0 if success else 1)