#!/usr/bin/env python3
"""
Quick tuning with real-time progress monitoring
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path
import json

def monitor_progress(results_dir, stop_event):
    """Monitor tuning progress in real time"""
    while not stop_event.is_set():
        try:
            # Check if results file exists
            results_file = Path(results_dir) / "tuning_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    data = json.load(f)

                n_trials = data.get('n_trials', 0)
                best_score = data.get('best_score', -1.0)

                if n_trials > 0:
                    print(f"🔄 Progress: {n_trials}/5 trials completed")
                    if best_score > 0:
                        print(f"📊 Current best accuracy: {best_score:.4f}")
                    else:
                        print(f"📊 Current best accuracy: Not found yet")

            # Check for log files
            log_files = list(Path("log_0909").glob("TF_DWT*.log")) if Path("log_0909").exists() else []
            if log_files:
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                mod_time = latest_log.stat().st_mtime
                if time.time() - mod_time < 60:  # Updated in last minute
                    print(f"🏃 Active training detected in: {latest_log.name}")

            time.sleep(30)  # Check every 30 seconds

        except Exception as e:
            print(f"⚠️ Monitoring error: {e}")
            time.sleep(30)

def run_with_progress():
    """Run quick tuning with progress monitoring"""
    results_dir = "quick_progress_results"

    print("🚀 Starting TF-DWT hyperparameter tuning (5 trials)")
    print("⏱️  Estimated time: 1.5-3 hours")
    print("📈 Each trial runs nested cross-validation (5 folds x 5 repeats)")
    print("=" * 60)

    # Start progress monitor thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_progress, args=(results_dir, stop_event))
    monitor_thread.daemon = True
    monitor_thread.start()

    try:
        # Run the tuning
        cmd = [
            sys.executable, "tune_tfdwt.py",
            "--strategy", "random",
            "--n_trials", "5",
            "--results_dir", results_dir
        ]

        print(f"🔧 Command: {' '.join(cmd)}")
        print("📝 Monitor logs with: tail -f log_0909/TF_DWT*.log")
        print("=" * 60)

        start_time = time.time()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Print output in real time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] {output.strip()}")

        # Get final result
        return_code = process.poll()
        stderr = process.stderr.read()

        total_time = time.time() - start_time

        print("=" * 60)
        print(f"🏁 Process completed in {total_time/60:.1f} minutes")
        print(f"📋 Return code: {return_code}")

        if return_code == 0:
            print("✅ Tuning completed successfully!")

            # Show final results
            results_file = Path(results_dir) / "tuning_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    data = json.load(f)

                best_score = data.get('best_score', -1.0)
                best_params = data.get('best_params', {})

                print(f"🎯 Best accuracy: {best_score:.4f}")
                if best_params:
                    print("🔧 Best parameters:")
                    for key, value in best_params.items():
                        if isinstance(value, float):
                            print(f"  {key}: {value:.4f}")
                        else:
                            print(f"  {key}: {value}")
            else:
                print("⚠️ Results file not found")
        else:
            print("❌ Tuning failed!")
            if stderr:
                print("Error output:")
                print(stderr[:1000])  # First 1000 chars

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
        process.terminate()
        return_code = -1

    except Exception as e:
        print(f"❌ Error: {e}")
        return_code = -1

    finally:
        stop_event.set()
        monitor_thread.join(timeout=1)

    return return_code

if __name__ == "__main__":
    exit_code = run_with_progress()
    sys.exit(exit_code)