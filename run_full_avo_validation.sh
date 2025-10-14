#!/bin/bash
# Full AVO Scenario Validation (5 runs)
# Config: P3=80, AVO=10, Target: AVO ≥ 0.66

echo "=== AVO SCENARIO FULL VALIDATION ==="
echo "Configuration:"
echo "  P3 trials/subject: 80"
echo "  AVO trials/subject: 10"
echo "  Target: AVO accuracy ≥ 0.66 for ALL 5 runs"
echo ""

# Verify config
p3_trials=$(grep "NESTED_CV_TRIALS_PER_SUBJECT_P3 =" config.py | grep -v "#" | awk -F= '{print $2}' | tr -d ' ')
avo_trials=$(grep "NESTED_CV_TRIALS_PER_SUBJECT_AVO =" config.py | grep -v "#" | awk -F= '{print $2}' | tr -d ' ')

if [ "$p3_trials" != "80" ] || [ "$avo_trials" != "10" ]; then
    echo "ERROR: Config mismatch!"
    echo "Expected: P3=80, AVO=10"
    echo "Found: P3=$p3_trials, AVO=$avo_trials"
    echo "Please update config.py manually."
    exit 1
fi

echo "✓ Config verified: P3=$p3_trials, AVO=$avo_trials"
echo ""

results_file="avo_validation_results.txt"
echo "=== AVO Validation Results ===" > $results_file
echo "Start time: $(date)" >> $results_file
echo "" >> $results_file

for i in {1..5}; do
    echo "===================="
    echo "=== Run $i/5 ==="
    echo "===================="
    echo "Start: $(date)"

    conda run -n eeg python main_tfdwt.py 2>&1 | tee "avo_val_run${i}.log"

    # Extract results
    latest_csv=$(ls -t tfdwt_summary_stats_*.csv 2>/dev/null | head -1)
    if [ -f "$latest_csv" ]; then
        avo=$(tail -1 "$latest_csv" | cut -d',' -f9)
        p3=$(tail -1 "$latest_csv" | cut -d',' -f5)

        echo ""
        echo "--- Run $i Results ---"
        echo "AVO: $avo"
        echo "P3:  $p3"

        # Log to file
        echo "Run $i: AVO=$avo, P3=$p3" >> $results_file

        # Check target
        if (( $(echo "$avo >= 0.66" | bc -l) )); then
            echo "✓ TARGET MET"
            echo "Run $i: ✓ PASS" >> $results_file
        else
            gap=$(echo "0.66 - $avo" | bc -l)
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

# Extract all AVO results
avo_results=$(grep "Run [0-9]: AVO=" $results_file | cut -d'=' -f2 | cut -d',' -f1)
pass_count=$(grep "✓ PASS" $results_file | wc -l)

echo "Results:" | tee -a $results_file
i=1
for avo in $avo_results; do
    status="✗"
    if (( $(echo "$avo >= 0.66" | bc -l) )); then
        status="✓"
    fi
    echo "  Run $i: $avo $status" | tee -a $results_file
    i=$((i+1))
done

echo "" | tee -a $results_file
echo "Success rate: $pass_count/5" | tee -a $results_file

if [ "$pass_count" -eq 5 ]; then
    echo "" | tee -a $results_file
    echo "🎉 AVO SCENARIO VALIDATION PASSED! 🎉" | tee -a $results_file
    echo "All 5 runs achieved AVO ≥ 0.66" | tee -a $results_file
else
    echo "" | tee -a $results_file
    echo "⚠ AVO SCENARIO VALIDATION FAILED" | tee -a $results_file
    echo "Only $pass_count/5 runs passed" | tee -a $results_file
fi

echo "" | tee -a $results_file
echo "End time: $(date)" | tee -a $results_file

cat $results_file
