#!/bin/bash
echo "=== Round 5: FINE-TUNED Enhancement (方案A) ==="
echo "Settings:"
echo "  - Weight multiplier: 2.0x → cap at 17x (was 1.75x → 15x)"
echo "  - MMD: 0.68 for high ratio (was 0.65)"
echo "  - Focal: gamma=3.6, alpha=0.65 (was 3.5, 0.6)"
echo "  - Batch/LR/Epochs: Keep from Round 4"
echo ""
echo "Goal: 连续5次 AVO ≥ 0.66"
echo ""

for i in {1..5}; do
  echo "=== Run $i/5 ==="
  echo "Start: $(date)"

  conda run -n eeg python main_tfdwt.py 2>&1 | tee "round5_run${i}.log"

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

echo "=== Round 5 完成 ==="
echo "分析所有5次结果..."
