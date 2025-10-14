#!/bin/bash
# Wait for experiment completion and notify

LOG_FILE="log_0909/TF_DWT_results_20251003_003630.log"
CHECK_INTERVAL=60  # Check every 60 seconds

echo "=== Monitoring experiment for completion ==="
echo "Log file: $LOG_FILE"
echo "Start time: $(date)"
echo ""

while true; do
    # Check if CSV result file was created
    latest_csv=$(ls -t tfdwt_summary_stats_20251003*.csv 2>/dev/null | head -1)

    if [ -n "$latest_csv" ]; then
        echo ""
        echo "=========================================="
        echo "✓ EXPERIMENT COMPLETED!"
        echo "=========================================="
        echo "Completion time: $(date)"
        echo "Result file: $latest_csv"
        echo ""

        # Extract results
        avo=$(tail -1 "$latest_csv" | cut -d',' -f9)
        p3=$(tail -1 "$latest_csv" | cut -d',' -f5)

        echo "=== RESULTS ==="
        echo "AVO accuracy: $avo"
        echo "P3 accuracy:  $p3"
        echo ""

        # Check target
        if (( $(echo "$avo >= 0.66" | bc -l) )); then
            echo "🎉 AVO TARGET MET! (≥0.66)"
        else
            gap=$(echo "0.66 - $avo" | bc -l)
            echo "⚠ AVO below target. Gap: $gap"
        fi

        echo ""
        echo "Full results saved to: $latest_csv"
        break
    fi

    # Show current progress
    tail_output=$(tail -3 "$LOG_FILE" 2>/dev/null)
    current_epoch=$(echo "$tail_output" | grep "Epoch" | tail -1 | grep -oP "Epoch \d+/\d+" | tail -1)
    current_avo=$(echo "$tail_output" | grep "Val(AVO)" | tail -1 | grep -oP "Val\(AVO\)=[0-9.]+")
    patience=$(echo "$tail_output" | grep "patience remaining" | tail -1 | grep -oP "patience remaining: \d+/\d+" | tail -1)

    if [ -n "$current_epoch" ]; then
        echo "[$(date +%H:%M:%S)] $current_epoch | $current_avo | $patience"
    fi

    sleep $CHECK_INTERVAL
done

echo ""
echo "Monitoring complete."
