#!/bin/bash
# Monitor experiment progress

echo "=== Experiment Progress Monitor ==="
echo "Current time: $(date)"
echo ""

# Check running processes
running=$(ps aux | grep "python main_tfdwt" | grep -v grep | wc -l)
if [ $running -gt 0 ]; then
    echo "✓ Experiment running ($running processes)"
    ps aux | grep "python main_tfdwt" | grep -v grep | head -2
else
    echo "✗ No experiment running"
fi

echo ""
echo "=== Latest Results ==="

# Find latest CSV
latest=$(ls -t tfdwt_summary_stats_*.csv 2>/dev/null | head -1)
if [ -f "$latest" ]; then
    timestamp=$(echo $latest | grep -oP '\d{8}_\d{6}')
    echo "Latest result: $timestamp"

    avo=$(tail -1 "$latest" | cut -d',' -f9)
    p3=$(tail -1 "$latest" | cut -d',' -f5)

    echo "  AVO accuracy: $avo"
    echo "  P3 accuracy: $p3"

    # Check targets
    if (( $(echo "$avo >= 0.66" | bc -l) )); then
        echo "  AVO: ✓ ≥0.66"
    else
        gap=$(echo "0.66 - $avo" | bc -l)
        echo "  AVO: ✗ gap=$gap"
    fi

    if (( $(echo "$p3 >= 0.62" | bc -l) )); then
        echo "  P3: ✓ ≥0.62"
    else
        gap=$(echo "0.62 - $p3" | bc -l)
        echo "  P3: ✗ gap=$gap"
    fi
else
    echo "No results yet"
fi

echo ""
echo "=== Recent Result Files ==="
ls -lth tfdwt_summary_stats_*.csv 2>/dev/null | head -5

echo ""
echo "=== Log Files ==="
ls -lth prototype_test_*.log avo_val_*.log p3_val_*.log 2>/dev/null | head -5
