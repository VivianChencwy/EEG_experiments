#!/bin/bash
# Prototype-based TF-DWT Test Run
# New strategy: Mixup + Prototype Networks + Shared BN

echo "=== PROTOTYPE-BASED TF-DWT TEST ==="
echo ""
echo "New Strategy:"
echo "  1. Mixup (α=0.4) for small domain data augmentation"
echo "  2. Prototype networks for discriminative transfer"
echo "  3. Shared BN statistics (no split-BN)"
echo "  4. Conservative weights (sqrt scaling, max 12x)"
echo "  5. Reduced MMD (0.2-0.4, down from 0.65-0.68)"
echo "  6. Prototype loss (λ=0.5-0.8)"
echo ""
echo "Expected improvements:"
echo "  - Better small-domain generalization via Mixup"
echo "  - Discriminative transfer via prototypes"
echo "  - Preserve large-domain performance"
echo ""
echo "Starting single test run..."
date

conda run -n eeg python main_tfdwt.py 2>&1 | tee prototype_test_run1.log

latest_csv=$(ls -t tfdwt_summary_stats_*.csv 2>/dev/null | head -1)
if [ -f "$latest_csv" ]; then
    echo ""
    echo "=== RESULTS ==="
    avo=$(tail -1 "$latest_csv" | cut -d',' -f9)
    p3=$(tail -1 "$latest_csv" | cut -d',' -f5)
    echo "AVO accuracy: $avo"
    echo "P3 accuracy: $p3"

    # Check if AVO meets target
    if (( $(echo "$avo >= 0.66" | bc -l) )); then
        echo "✓ AVO TARGET MET (≥0.66)"
    else:
        gap=$(echo "0.66 - $avo" | bc -l)
        echo "✗ AVO below target. Gap: $gap"
    fi

    # Check P3 not degraded too much
    if (( $(echo "$p3 >= 0.55" | bc -l) )); then
        echo "✓ P3 performance acceptable (≥0.55)"
    else
        echo "⚠ P3 performance degraded: $p3"
    fi
fi

echo ""
date
echo "Test run complete."
