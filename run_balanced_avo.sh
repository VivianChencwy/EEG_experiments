#!/bin/bash
echo "=== Round 4: BALANCED Optimization for Stability ==="
echo "Settings:"
echo "  - Weight: 15x (was 20x)"
echo "  - MMD: 0.65 (was 0.8)"
echo "  - Focal: gamma=3.5, alpha=0.6 (was 4.0, 0.7)"
echo "  - Batch/LR/Epochs: Keep effective settings from Round 3"
echo ""

for i in {1..5}; do
  echo "=== Run $i/5 ==="
  echo "Start: $(date)"
  
  conda run -n eeg python main_tfdwt.py 2>&1 | tee "balanced_avo_run${i}.log"
  
  latest_csv=$(ls -t tfdwt_summary_stats_*.csv 2>/dev/null | head -1)
  if [ -f "$latest_csv" ]; then
    avo=$(tail -1 "$latest_csv" | cut -d',' -f9)
    echo "AVO: $avo"
    
    if (( $(echo "$avo >= 0.66" | bc -l) )); then
      echo "✓ TARGET MET"
    else
      gap=$(echo "0.66 - $avo" | bc -l)
      echo "✗ Gap: $gap"
    fi
  fi
  
  echo "End: $(date)"
  echo ""
done
