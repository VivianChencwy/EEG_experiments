#!/bin/bash

# Update config for P3 scenario
python3 << PYTHON
import re
config_path = '/home/vivian/eeg/EEG_experiments/config.py'
with open(config_path, 'r') as f:
    content = f.read()

content = re.sub(r'NESTED_CV_TRIALS_PER_SUBJECT_P3\s*=\s*\d+', 
                 'NESTED_CV_TRIALS_PER_SUBJECT_P3 = 10', content)
content = re.sub(r'NESTED_CV_TRIALS_PER_SUBJECT_AVO\s*=\s*\d+', 
                 'NESTED_CV_TRIALS_PER_SUBJECT_AVO = 80', content)

with open(config_path, 'w') as f:
    f.write(content)
print("Config updated: P3=10, AVO=80")
PYTHON

for i in {1..5}; do
  echo "=== EXTREME Optimization - P3 - Run $i/5 ==="
  echo "Start: $(date)"
  
  conda run -n eeg python main_tfdwt.py 2>&1 | tee "extreme_p3_run${i}.log"
  
  latest_csv=$(ls -t tfdwt_summary_stats_*.csv 2>/dev/null | head -1)
  if [ -f "$latest_csv" ]; then
    p3=$(tail -1 "$latest_csv" | cut -d',' -f5)
    echo "P3 Result: $p3"
    
    if (( $(echo "$p3 >= 0.62" | bc -l) )); then
      echo "✓ TARGET MET!"
    else
      gap=$(echo "0.62 - $p3" | bc -l)
      echo "✗ Gap: $gap"
    fi
  fi
  
  echo "End: $(date)"
  echo ""
done
