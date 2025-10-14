#!/bin/bash
# Automated P3 scenario experiments (P3=10, AVO=80)

echo "=== P3 Scenario Experiments ==="
echo "Configuration: P3=10, AVO=80"
echo "Target: P3 accuracy >= 0.62 for 5 consecutive runs"
echo ""

# Update config for P3 scenario
python3 << PYTHON
config_path = '/home/vivian/eeg/EEG_experiments/config.py'
with open(config_path, 'r') as f:
    content = f.read()

# Replace trials configuration
import re
content = re.sub(r'NESTED_CV_TRIALS_PER_SUBJECT_P3\s*=\s*\d+', 'NESTED_CV_TRIALS_PER_SUBJECT_P3 = 10', content)
content = re.sub(r'NESTED_CV_TRIALS_PER_SUBJECT_AVO\s*=\s*\d+', 'NESTED_CV_TRIALS_PER_SUBJECT_AVO = 80', content)

with open(config_path, 'w') as f:
    f.write(content)

print("Updated config: P3=10, AVO=80")
PYTHON

# Run 5 experiments
for i in {1..5}; do
  echo "=== P3 Experiment $i/5 ==="
  echo "Start: $(date)"
  
  conda run -n eeg python main_tfdwt.py > "p3_scenario_run${i}_$(date +%Y%m%d_%H%M%S).log" 2>&1
  
  # Extract results
  latest_csv=$(ls -t tfdwt_summary_stats_*.csv 2>/dev/null | head -1)
  if [ -f "$latest_csv" ]; then
    p3=$(tail -1 "$latest_csv" | cut -d',' -f5)
    avo=$(tail -1 "$latest_csv" | cut -d',' -f9)
    echo "Results - P3: $p3, AVO: $avo"
    
    if (( $(echo "$p3 >= 0.62" | bc -l) )); then
      echo "✓ P3 target met!"
    else
      echo "✗ P3 below target"
    fi
  fi
  
  echo "End: $(date)"
  echo ""
done

echo "=== All P3 experiments completed ==="
