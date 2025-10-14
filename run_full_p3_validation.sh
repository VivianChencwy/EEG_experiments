#!/bin/bash
# Full P3 Scenario Validation (5 runs)
# Config: P3=10, AVO=80, Target: P3 ≥ 0.62

echo "=== P3 SCENARIO FULL VALIDATION ==="
echo "Configuration:"
echo "  P3 trials/subject: 10"
echo "  AVO trials/subject: 80"
echo "  Target: P3 accuracy ≥ 0.62 for ALL 5 runs"
echo ""

# Verify config
p3_trials=$(grep "NESTED_CV_TRIALS_PER_SUBJECT_P3 =" config.py | grep -v "#" | awk -F= '{print $2}' | tr -d ' ')
avo_trials=$(grep "NESTED_CV_TRIALS_PER_SUBJECT_AVO =" config.py | grep -v "#" | awk -F= '{print $2}' | tr -d ' ')

if [ "$p3_trials" != "10" ] || [ "$avo_trials" != "80" ]; then
    echo "ERROR: Config mismatch!"
    echo "Expected: P3=10, AVO=80"
    echo "Found: P3=$p3_trials, AVO=$avo_trials"
    echo "Please update config.py manually."
    exit 1
fi

echo "✓ Config verified: P3=$p3_trials, AVO=$avo_trials"
echo ""

results_file="p3_validation_results.txt"
echo "=== P3 Validation Results ===" > $results_file
echo "Start time: $(date)" >> $results_file
echo "" >> $results_file

for i in {1..5}; do
    echo "===================="
    echo "=== Run $i/5 ==="
    echo "===================="
    echo "Start: $(date)"

    conda run -n eeg python main_tfdwt.py 2>&1 | tee "p3_val_run${i}.log"

    # Extract results
    latest_csv=$(ls -t tfdwt_summary_stats_*.csv 2>/dev/null | head -1)
    if [ -f "$latest_csv" ]; then
        p3=$(tail -1 "$latest_csv" | cut -d',' -f5)
        avo=$(tail -1 "$latest_csv" | cut -d',' -f9)

        echo ""
        echo "--- Run $i Results ---"
        echo "P3:  $p3"
        echo "AVO: $avo"

        # Log to file
        echo "Run $i: P3=$p3, AVO=$avo" >> $results_file

        # Check target
        if (( $(echo "$p3 >= 0.62" | bc -l) )); then
            echo "✓ TARGET MET"
            echo "Run $i: ✓ PASS" >> $results_file
        else
            gap=$(echo "0.62 - $p3" | bc -l)
            echo "✗ FAILED - Gap: $gap"
            echo "Run $i: ✗ FAIL (gap=$gap)" >> $results_file
        fi
    else
        echo "ERROR: No result file found!"
        echo "Run $i: ERROR - No results" >> $results_file
    fi

    echo "End: $(date)"
    echo ""
done

# Summary
echo "======================================" | tee -a $results_file
echo "=== VALIDATION SUMMARY ===" | tee -a $results_file
echo "======================================" | tee -a $results_file
echo "" | tee -a $results_file

# Extract all P3 results
p3_results=$(grep "Run [0-9]: P3=" $results_file | cut -d'=' -f2 | cut -d',' -f1)
pass_count=$(grep "✓ PASS" $results_file | wc -l)

echo "Results:" | tee -a $results_file
i=1
for p3 in $p3_results; do
    status="✗"
    if (( $(echo "$p3 >= 0.62" | bc -l) )); then
        status="✓"
    fi
    echo "  Run $i: $p3 $status" | tee -a $results_file
    i=$((i+1))
done

echo "" | tee -a $results_file
echo "Success rate: $pass_count/5" | tee -a $results_file

if [ "$pass_count" -eq 5 ]; then
    echo "" | tee -a $results_file
    echo "🎉 P3 SCENARIO VALIDATION PASSED! 🎉" | tee -a $results_file
    echo "All 5 runs achieved P3 ≥ 0.62" | tee -a $results_file
else
    echo "" | tee -a $results_file
    echo "⚠ P3 SCENARIO VALIDATION FAILED" | tee -a $results_file
    echo "Only $pass_count/5 runs passed" | tee -a $results_file
fi

echo "" | tee -a $results_file
echo "End time: $(date)" | tee -a $results_file

cat $results_file
