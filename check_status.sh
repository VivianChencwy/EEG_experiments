#!/bin/bash
# Check status of automated experiments

echo "===== Experiment Status Check ====="
echo "Time: $(date)"
echo ""

# Check running processes
echo "=== Running Processes ==="
ps aux | grep -E "(main_tfdwt|monitor_and_run)" | grep -v grep | awk '{print "PID "$2": "$11" "$12" "$13" "$14" "$15}'
echo ""

# Check latest results
echo "=== Latest TF-DWT Results ==="
latest_csv=$(ls -t tfdwt_summary_stats_*.csv 2>/dev/null | head -1)
if [ -f "$latest_csv" ]; then
    echo "File: $latest_csv"
    echo "Modified: $(stat -c %y "$latest_csv" | cut -d' ' -f1-2)"

    # Extract key metrics (correct column numbers)
    overall_acc=$(tail -1 "$latest_csv" | cut -d',' -f1)
    p3_acc=$(tail -1 "$latest_csv" | cut -d',' -f5)
    avo_acc=$(tail -1 "$latest_csv" | cut -d',' -f9)

    echo "P3 Accuracy: $p3_acc"
    echo "AVO Accuracy: $avo_acc"
    echo "Overall Accuracy: $overall_acc"
else
    echo "No results CSV found yet"
fi

echo ""

# Count total results
echo "=== Total Experiments Completed ==="
ls tfdwt_summary_stats_*.csv 2>/dev/null | wc -l

echo ""

# Check logs
echo "=== Recent Log Files ==="
ls -lht *.log 2>/dev/null | head -5

echo ""
echo "===== End of Status Check ====="
