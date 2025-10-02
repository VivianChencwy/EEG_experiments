#!/bin/bash
# Quick progress checker for TF-DWT optimization

echo "==================================================="
echo "TF-DWT Optimization Progress Check"
echo "Time: $(date)"
echo "==================================================="
echo ""

# Check running processes
echo "📊 Running Experiments:"
ps aux | grep "main_tfdwt_v" | grep -v grep | while read -r line; do
    pid=$(echo $line | awk '{print $2}')
    cmd=$(echo $line | awk '{print $11}')
    time=$(echo $line | awk '{print $10}')
    version=$(echo $cmd | grep -oP 'v\d+_[a-z0-9_]+' | head -1)
    echo "  ✓ $version (PID: $pid, Runtime: $time)"
done

if ! ps aux | grep "main_tfdwt_v" | grep -v grep > /dev/null; then
    echo "  ⚠️  No experiments currently running"
fi

echo ""

# Check for completed results
echo "📁 Completed Results:"
for csv in tfdwt_v*_detailed_*.csv; do
    if [ -f "$csv" ]; then
        version=$(echo $csv | grep -oP 'v\d+_[a-z0-9_]+')
        avo_acc=$(python3 -c "
import pandas as pd
try:
    df = pd.read_csv('$csv')
    print(f\"{df['avo_accuracy'].mean():.4f} ± {df['avo_accuracy'].std():.4f}\")
except:
    print('ERROR')
" 2>/dev/null)

        if [ "$avo_acc" != "ERROR" ]; then
            # Check if target achieved
            target_check=$(python3 -c "
import pandas as pd
df = pd.read_csv('$csv')
print('✅' if df['avo_accuracy'].mean() >= 0.65 else '  ')
" 2>/dev/null)

            echo "  $target_check $version: AVO Accuracy = $avo_acc"
        fi
    fi
done

if ! ls tfdwt_v*_detailed_*.csv > /dev/null 2>&1; then
    echo "  ⚠️  No results found yet"
fi

echo ""

# Check latest log for errors
echo "🔍 Latest Log Activity:"
latest_log=$(ls -t log_0909/tfdwt_v*_*.log 2>/dev/null | head -1)
if [ -n "$latest_log" ]; then
    echo "  Log: $(basename $latest_log)"
    echo "  Last 3 lines:"
    tail -3 "$latest_log" | sed 's/^/    /'
else
    echo "  ⚠️  No logs found"
fi

echo ""
echo "==================================================="
echo "Commands:"
echo "  Watch V1 log:  tail -f log_0909/tfdwt_v1_*.log"
echo "  Kill V1:       pkill -f main_tfdwt_v1"
echo "  Run V2:        python main_tfdwt_v2_cap6_mmd01_warm20.py"
echo "  Run V3:        python main_tfdwt_v3_cap8_mmd01_noguard.py"
echo "  Auto-run all:  ./run_optimization_experiments.sh"
echo "==================================================="
