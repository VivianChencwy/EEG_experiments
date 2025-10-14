#!/bin/bash
# Automatic Full Validation Master Script
# Runs all tests automatically and generates final report

MASTER_LOG="auto_validation_master.log"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "=================================================="
echo "=== AUTOMATIC FULL VALIDATION MASTER SCRIPT ===="
echo "=================================================="
echo "Start time: $(date)"
echo ""

#----------------------------------------------------
# PHASE 1: Initial Test
#----------------------------------------------------
echo "###  PHASE 1: Initial Prototype Test ###"
echo ""

# Wait for current test to finish if running
while ps aux | grep -v grep | grep "python main_tfdwt" > /dev/null; do
    echo "Waiting for current test to finish..."
    sleep 60
done

# Check if we already have a test result
latest=$(ls -t tfdwt_summary_stats_*.csv 2>/dev/null | head -1)
if [ -f "$latest" ]; then
    timestamp=$(echo $latest | grep -oP '\d{8}_\d{6}')
    file_time=$(stat -c %Y "$latest")
    current_time=$(date +%s)
    age=$((current_time - file_time))

    # If result is less than 1 hour old, use it
    if [ $age -lt 3600 ]; then
        echo "✓ Using recent test result: $timestamp ($(($age/60)) minutes old)"
        avo=$(tail -1 "$latest" | cut -d',' -f9)
        p3=$(tail -1 "$latest" | cut -d',' -f5)

        echo "Phase 1 Result: AVO=$avo, P3=$p3"

        if (( $(echo "$avo >= 0.66" | bc -l) )); then
            echo "✓ Phase 1 PASSED - AVO ≥ 0.66"
            PHASE1_PASS=true
        else
            echo "✗ Phase 1 FAILED - AVO < 0.66"
            PHASE1_PASS=false
        fi
    fi
fi

# If no recent result, the current test should finish soon
if [ -z "$PHASE1_PASS" ]; then
    echo "Waiting for Phase 1 test to complete..."
    while ! ls -t tfdwt_summary_stats_*.csv 2>/dev/null | head -1 | grep -q "$(date +%Y%m%d)"; do
        sleep 30
    done

    latest=$(ls -t tfdwt_summary_stats_*.csv 2>/dev/null | head -1)
    avo=$(tail -1 "$latest" | cut -d',' -f9)
    p3=$(tail -1 "$latest" | cut -d',' -f5)

    echo "Phase 1 Result: AVO=$avo, P3=$p3"

    if (( $(echo "$avo >= 0.66" | bc -l) )); then
        echo "✓ Phase 1 PASSED"
        PHASE1_PASS=true
    else
        echo "✗ Phase 1 FAILED"
        PHASE1_PASS=false
    fi
fi

echo ""
echo "Phase 1 complete: $(date)"
echo ""

if [ "$PHASE1_PASS" = false ]; then
    echo "=================================================="
    echo "⚠ VALIDATION ABORTED - Phase 1 failed"
    echo "=================================================="
    echo ""
    echo "Phase 1 test did not meet AVO ≥ 0.66 target."
    echo "Please review and adjust hyperparameters."
    exit 1
fi

#----------------------------------------------------
# PHASE 2: AVO Scenario (5 runs)
#----------------------------------------------------
echo "###  PHASE 2: AVO Scenario Validation (5 runs) ###"
echo ""

# Ensure config is correct
echo "Setting config for AVO scenario..."
sed -i 's/^NESTED_CV_TRIALS_PER_SUBJECT_P3 = .*/NESTED_CV_TRIALS_PER_SUBJECT_P3 = 80/' config.py
sed -i 's/^NESTED_CV_TRIALS_PER_SUBJECT_AVO = .*/NESTED_CV_TRIALS_PER_SUBJECT_AVO = 10/' config.py

p3_val=$(grep "NESTED_CV_TRIALS_PER_SUBJECT_P3 =" config.py | grep -v "#" | awk -F= '{print $2}' | tr -d ' ')
avo_val=$(grep "NESTED_CV_TRIALS_PER_SUBJECT_AVO =" config.py | grep -v "#" | awk -F= '{print $2}' | tr -d ' ')

echo "Config: P3=$p3_val, AVO=$avo_val"

if [ "$p3_val" != "80" ] || [ "$avo_val" != "10" ]; then
    echo "ERROR: Failed to set config!"
    exit 1
fi

echo "✓ Config confirmed"
echo ""

# Run AVO validation
./run_full_avo_validation.sh

# Check if all passed
if grep -q "🎉 AVO SCENARIO VALIDATION PASSED" avo_validation_results.txt; then
    echo "✓ Phase 2 PASSED"
    PHASE2_PASS=true
else
    echo "✗ Phase 2 FAILED"
    PHASE2_PASS=false
fi

echo ""
echo "Phase 2 complete: $(date)"
echo ""

if [ "$PHASE2_PASS" = false ]; then
    echo "=================================================="
    echo "⚠ VALIDATION INCOMPLETE - Phase 2 failed"
    echo "=================================================="
    echo ""
    echo "AVO scenario validation did not achieve 5/5 success."
    echo "Results saved in avo_validation_results.txt"
    echo ""
    echo "Proceeding to Phase 3 for comparison..."
    echo ""
fi

#----------------------------------------------------
# PHASE 3: P3 Scenario (5 runs)
#----------------------------------------------------
echo "###  PHASE 3: P3 Scenario Validation (5 runs) ###"
echo ""

# Set config for P3 scenario
echo "Setting config for P3 scenario..."
sed -i 's/^NESTED_CV_TRIALS_PER_SUBJECT_P3 = .*/NESTED_CV_TRIALS_PER_SUBJECT_P3 = 10/' config.py
sed -i 's/^NESTED_CV_TRIALS_PER_SUBJECT_AVO = .*/NESTED_CV_TRIALS_PER_SUBJECT_AVO = 80/' config.py

p3_val=$(grep "NESTED_CV_TRIALS_PER_SUBJECT_P3 =" config.py | grep -v "#" | awk -F= '{print $2}' | tr -d ' ')
avo_val=$(grep "NESTED_CV_TRIALS_PER_SUBJECT_AVO =" config.py | grep -v "#" | awk -F= '{print $2}' | tr -d ' ')

echo "Config: P3=$p3_val, AVO=$avo_val"

if [ "$p3_val" != "10" ] || [ "$avo_val" != "80" ]; then
    echo "ERROR: Failed to set config!"
    exit 1
fi

echo "✓ Config confirmed"
echo ""

# Run P3 validation
./run_full_p3_validation.sh

# Check if all passed
if grep -q "🎉 P3 SCENARIO VALIDATION PASSED" p3_validation_results.txt; then
    echo "✓ Phase 3 PASSED"
    PHASE3_PASS=true
else
    echo "✗ Phase 3 FAILED"
    PHASE3_PASS=false
fi

echo ""
echo "Phase 3 complete: $(date)"
echo ""

#----------------------------------------------------
# FINAL SUMMARY
#----------------------------------------------------
echo "=================================================="
echo "===  FINAL VALIDATION SUMMARY  ==="
echo "=================================================="
echo ""

echo "Phase 1 (Initial Test): $( [ "$PHASE1_PASS" = true ] && echo '✓ PASS' || echo '✗ FAIL' )"
echo "Phase 2 (AVO Scenario): $( [ "$PHASE2_PASS" = true ] && echo '✓ PASS' || echo '✗ FAIL' )"
echo "Phase 3 (P3 Scenario):  $( [ "$PHASE3_PASS" = true ] && echo '✓ PASS' || echo '✗ FAIL' )"

echo ""

if [ "$PHASE2_PASS" = true ] && [ "$PHASE3_PASS" = true ]; then
    echo "🎉🎉🎉 ALL VALIDATIONS PASSED! 🎉🎉🎉"
    echo ""
    echo "Both scenarios achieved 100% success rate!"
    echo "  - AVO scenario: 5/5 runs ≥ 0.66"
    echo "  - P3 scenario: 5/5 runs ≥ 0.62"
elif [ "$PHASE2_PASS" = true ]; then
    echo "✓ AVO Scenario validated successfully"
    echo "✗ P3 Scenario needs improvement"
elif [ "$PHASE3_PASS" = true ]; then
    echo "✗ AVO Scenario needs improvement"
    echo "✓ P3 Scenario validated successfully"
else
    echo "⚠ Both scenarios need further tuning"
fi

echo ""
echo "=================================================="
echo "End time: $(date)"
echo "=================================================="

# Generate final report
echo ""
echo "Generating final report..."
python3 -c "
import pandas as pd
import glob
from datetime import datetime

print('\n=== Detailed Results Analysis ===\n')

# Find all result CSVs from today
today = datetime.now().strftime('%Y%m%d')
all_csvs = sorted(glob.glob(f'tfdwt_summary_stats_{today}_*.csv'))

if all_csvs:
    print(f'Found {len(all_csvs)} results from today:\n')
    for csv in all_csvs[-10:]:  # Last 10
        df = pd.read_csv(csv)
        timestamp = csv.split('_')[-1].replace('.csv', '')
        avo = df['avo_accuracy_mean'].values[0]
        p3 = df['p3_accuracy_mean'].values[0]
        print(f'{timestamp}: AVO={avo:.4f}, P3={p3:.4f}')
else:
    print('No results found from today')
"

echo ""
echo "Full logs saved to: $MASTER_LOG"
echo "AVO results: avo_validation_results.txt"
echo "P3 results: p3_validation_results.txt"
