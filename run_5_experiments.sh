#!/bin/bash
# Run 5 experiments sequentially and save results

echo "===== Running 5 Sequential TF-DWT Experiments ====="
echo "Config: P3=80, AVO=10 (AVO-focused scenario)"
echo ""

# Arrays to store results
declare -a p3_accs
declare -a avo_accs
declare -a overall_accs

# Run 5 experiments
for i in {1..5}; do
    echo "===== Experiment $i/5 ====="
    echo "Start time: $(date)"

    # Run experiment
    conda run -n eeg python main_tfdwt.py > "run${i}_$(date +%Y%m%d_%H%M%S).log" 2>&1

    # Extract results from latest CSV
    latest_csv=$(ls -t tfdwt_summary_stats_*.csv | head -1)

    if [ -f "$latest_csv" ]; then
        # Extract accuracies (they're in specific columns)
        p3_acc=$(tail -1 "$latest_csv" | cut -d',' -f48)
        avo_acc=$(tail -1 "$latest_csv" | cut -d',' -f49)
        overall_acc=$(tail -1 "$latest_csv" | cut -d',' -f44)

        p3_accs+=("$p3_acc")
        avo_accs+=("$avo_acc")
        overall_accs+=("$overall_acc")

        echo "  P3 Accuracy: $p3_acc"
        echo "  AVO Accuracy: $avo_acc"
        echo "  Overall: $overall_acc"
    else
        echo "  ERROR: Could not find results CSV"
        p3_accs+=("N/A")
        avo_accs+=("N/A")
        overall_accs+=("N/A")
    fi

    echo "End time: $(date)"
    echo ""
done

# Summary
echo "===== SUMMARY OF 5 RUNS ====="
echo ""
echo "P3 Accuracies:"
for i in {0..4}; do
    echo "  Run $((i+1)): ${p3_accs[$i]}"
done

echo ""
echo "AVO Accuracies:"
for i in {0..4}; do
    echo "  Run $((i+1)): ${avo_accs[$i]}"
done

echo ""
echo "Overall Accuracies:"
for i in {0..4}; do
    echo "  Run $((i+1)): ${overall_accs[$i]}"
done

# Check if AVO target is met (all >= 0.66)
echo ""
echo "===== TARGET CHECK ====="
all_pass=true
for acc in "${avo_accs[@]}"; do
    if (( $(echo "$acc < 0.66" | bc -l) )); then
        all_pass=false
        break
    fi
done

if [ "$all_pass" = true ]; then
    echo "✓ SUCCESS! All 5 runs achieved AVO >= 0.66"
else
    echo "✗ Target not met. Some runs below 0.66"
fi

echo ""
echo "Results saved to run*.log files"
