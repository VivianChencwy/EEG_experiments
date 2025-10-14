#!/bin/bash
for i in {1..5}; do
  echo "=== EXTREME Optimization - AVO - Run $i/5 ==="
  echo "Start: $(date)"
  
  conda run -n eeg python main_tfdwt.py 2>&1 | tee "extreme_avo_run${i}.log"
  
  latest_csv=$(ls -t tfdwt_summary_stats_*.csv 2>/dev/null | head -1)
  if [ -f "$latest_csv" ]; then
    avo=$(tail -1 "$latest_csv" | cut -d',' -f9)
    echo "AVO Result: $avo"
    
    if (( $(echo "$avo >= 0.66" | bc -l) )); then
      echo "✓ TARGET MET!"
    else
      gap=$(echo "0.66 - $avo" | bc -l)
      echo "✗ Gap: $gap"
    fi
  fi
  
  echo "End: $(date)"
  echo ""
done
