#!/usr/bin/env python3
import time
import glob
import os
from datetime import datetime

LOG_FILE = "log_0909/TF_DWT_results_20251003_003630.log"
CHECK_INTERVAL = 60  # seconds

print("=== Monitoring experiment for completion ===")
print(f"Log file: {LOG_FILE}")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

last_size = 0
no_change_count = 0

while True:
    # Check for CSV result from today
    csv_files = glob.glob("tfdwt_summary_stats_20251003*.csv")

    if csv_files:
        latest_csv = max(csv_files, key=os.path.getmtime)
        print()
        print("=" * 50)
        print("✓ EXPERIMENT COMPLETED!")
        print("=" * 50)
        print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Result file: {latest_csv}")
        print()

        # Extract results
        with open(latest_csv, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 2:
                header = lines[0].strip().split(',')
                values = lines[1].strip().split(',')

                try:
                    avo_idx = header.index('avo_accuracy_mean')
                    p3_idx = header.index('p3_accuracy_mean')
                    avo = float(values[avo_idx])
                    p3 = float(values[p3_idx])

                    print("=== RESULTS ===")
                    print(f"AVO accuracy: {avo:.4f}")
                    print(f"P3 accuracy:  {p3:.4f}")
                    print()

                    if avo >= 0.66:
                        print("🎉 AVO TARGET MET! (≥0.66)")
                    else:
                        gap = 0.66 - avo
                        print(f"⚠ AVO below target. Gap: {gap:.4f}")

                except Exception as e:
                    print(f"Error parsing results: {e}")

        break

    # Check log file progress
    if os.path.exists(LOG_FILE):
        current_size = os.path.getsize(LOG_FILE)

        # Check if file stopped growing (possible completion or stuck)
        if current_size == last_size:
            no_change_count += 1
        else:
            no_change_count = 0

        last_size = current_size

        # Show current progress
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            recent = lines[-5:] if len(lines) >= 5 else lines

            for line in recent:
                if 'Epoch' in line and '1000' in line:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        epoch_info = parts[0].split('Epoch')[-1].strip() if 'Epoch' in parts[0] else ''
                        if epoch_info:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch_info}", end='')

                            # Look for AVO in next few lines
                            idx = lines.index(line)
                            for i in range(idx+1, min(idx+3, len(lines))):
                                if 'Val(AVO)' in lines[i]:
                                    import re
                                    match = re.search(r'Val\(AVO\)=([0-9.]+)', lines[i])
                                    if match:
                                        print(f" | AVO={match.group(1)}", end='')
                                if 'patience remaining' in lines[i]:
                                    import re
                                    match = re.search(r'patience remaining: (\d+/\d+)', lines[i])
                                    if match:
                                        print(f" | patience {match.group(1)}")
                                        break
                            else:
                                print()
                            break

        # If no change for 5 checks, might be stuck
        if no_change_count >= 5:
            print(f"\n⚠ Warning: Log file hasn't changed for {no_change_count * CHECK_INTERVAL} seconds")

    time.sleep(CHECK_INTERVAL)

print("\nMonitoring complete.")
