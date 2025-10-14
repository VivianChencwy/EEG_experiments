#!/bin/bash
# TF-DWT Optimization Experiment Manager
# Target: AVO accuracy >= 0.65
# Config: P3=80, AVO=10 (AVO is target)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="optimization_results"
mkdir -p $RESULTS_DIR

echo "=================================================="
echo "TF-DWT Optimization Experiments"
echo "Target: AVO Accuracy >= 0.65"
echo "Config: P3=80 trials, AVO=10 trials (AVO is target)"
echo "Start time: $(date)"
echo "=================================================="
echo ""

# Check configuration
echo "Verifying config.py..."
P3_TRIALS=$(grep "NESTED_CV_TRIALS_PER_SUBJECT_P3 =" config.py | grep -v "#" | awk '{print $3}')
AVO_TRIALS=$(grep "NESTED_CV_TRIALS_PER_SUBJECT_AVO =" config.py | grep -v "#" | awk '{print $3}')

if [ "$P3_TRIALS" != "80" ] || [ "$AVO_TRIALS" != "10" ]; then
    echo "ERROR: Config incorrect! Found P3=$P3_TRIALS, AVO=$AVO_TRIALS"
    echo "Expected: P3=80, AVO=10"
    exit 1
fi
echo "✓ Configuration correct: P3=80, AVO=10"
echo ""

# Function to extract AVO accuracy from CSV
get_avo_accuracy() {
    local csv_file=$1
    if [ -f "$csv_file" ]; then
        python3 -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('$csv_file')
    acc = df['avo_accuracy'].mean()
    std = df['avo_accuracy'].std()
    print(f'{acc:.4f}±{std:.4f}')
    sys.exit(0 if acc >= 0.65 else 1)
except:
    print('ERROR')
    sys.exit(2)
"
        return $?
    else
        echo "FILE_NOT_FOUND"
        return 2
    fi
}

# Function to run experiment
run_experiment() {
    local version=$1
    local name=$2
    local script=$3

    echo "=================================================="
    echo "[$version] $name"
    echo "Script: $script"
    echo "Start: $(date)"
    echo "=================================================="

    python $script > log_0909/${version}_${TIMESTAMP}.log 2>&1
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✓ Experiment completed successfully"

        # Find the generated CSV
        local csv_file=$(ls -t tfdwt_${version}_*_detailed_*.csv 2>/dev/null | head -1)

        if [ -n "$csv_file" ]; then
            # Get AVO accuracy
            local avo_acc=$(get_avo_accuracy "$csv_file")
            local acc_status=$?

            echo "  Results: $csv_file"
            echo "  AVO Accuracy: $avo_acc"

            # Move to results directory
            mv "$csv_file" "$RESULTS_DIR/"
            mv tfdwt_${version}_*_summary_*.csv "$RESULTS_DIR/" 2>/dev/null

            # Record result
            echo "$version,$name,$avo_acc,$csv_file" >> $RESULTS_DIR/experiment_log_${TIMESTAMP}.csv

            # Check if target achieved
            if [ $acc_status -eq 0 ]; then
                echo ""
                echo "🎉🎉🎉 TARGET ACHIEVED! 🎉🎉🎉"
                echo "Version: $version - $name"
                echo "AVO Accuracy: $avo_acc >= 0.65"
                echo ""

                # Save optimal configuration
                cat > $RESULTS_DIR/optimal_config_${version}.json << EOF
{
  "version": "$version",
  "name": "$name",
  "avo_accuracy": "$avo_acc",
  "csv_file": "$csv_file",
  "timestamp": "$(date -Iseconds)",
  "config": {
    "P3_trials": $P3_TRIALS,
    "AVO_trials": $AVO_TRIALS
  }
}
EOF
                echo "Optimal configuration saved to: $RESULTS_DIR/optimal_config_${version}.json"
                return 0  # Success - target achieved
            fi

            return 1  # Continue - target not achieved
        else
            echo "✗ No CSV file found"
            return 2  # Error
        fi
    else
        echo "✗ Experiment failed with exit code: $exit_code"
        return 2  # Error
    fi
}

# Initialize results log
echo "version,name,avo_accuracy,csv_file" > $RESULTS_DIR/experiment_log_${TIMESTAMP}.csv

# Run experiments sequentially until target is achieved
experiments=(
    "v1_cap8_mmd015:Version 1 (cap=8, mmd=0.15):main_tfdwt_v1_cap8_mmd015.py"
    "v2_cap6_mmd01_warm20:Version 2 (cap=6, mmd=0.1, warmup=20):main_tfdwt_v2_cap6_mmd01_warm20.py"
    "v3_cap8_mmd01_noguard:Version 3 (cap=8, mmd=0.1, no guard):main_tfdwt_v3_cap8_mmd01_noguard.py"
)

TARGET_ACHIEVED=false

for exp in "${experiments[@]}"; do
    IFS=':' read -r version name script <<< "$exp"

    echo ""
    run_experiment "$version" "$name" "$script"
    status=$?

    if [ $status -eq 0 ]; then
        TARGET_ACHIEVED=true
        break
    elif [ $status -eq 2 ]; then
        echo "⚠️  Experiment error, continuing to next..."
    fi

    echo ""
    echo "Waiting 10 seconds before next experiment..."
    sleep 10
done

# Summary
echo ""
echo "=================================================="
echo "OPTIMIZATION SUMMARY"
echo "=================================================="

if [ "$TARGET_ACHIEVED" = true ]; then
    echo "✅ TARGET ACHIEVED!"
    echo ""
    echo "Results saved in: $RESULTS_DIR/"
    ls -lh $RESULTS_DIR/optimal_config_*.json 2>/dev/null
else
    echo "⚠️  Target not achieved. Best results:"
    echo ""
    cat $RESULTS_DIR/experiment_log_${TIMESTAMP}.csv | column -t -s,
fi

echo ""
echo "End time: $(date)"
echo "All logs in: log_0909/"
echo "All results in: $RESULTS_DIR/"
echo "=================================================="
