#!/bin/bash
# Monitor ultra-aggressive experiments

while true; do
  clear
  echo "=== Ultra-Aggressive Experiments Monitor ==="
  echo "Time: $(date)"
  echo ""
  
  # Count completed experiments
  completed=$(ls ultra_aggressive_run*.log 2>/dev/null | wc -l)
  echo "Completed experiments: $completed/5"
  echo ""
  
  # Check if still running
  if ps aux | grep -q "[p]ython main_tfdwt.py"; then
    echo "Status: RUNNING"
    echo ""
    
    # Show latest results
    echo "=== Latest Results ==="
    latest_csv=$(ls -t tfdwt_summary_stats_*.csv 2>/dev/null | head -1)
    if [ -f "$latest_csv" ]; then
      echo "File: $(basename $latest_csv)"
      p3=$(tail -1 "$latest_csv" | cut -d',' -f5)
      avo=$(tail -1 "$latest_csv" | cut -d',' -f9)
      overall=$(tail -1 "$latest_csv" | cut -d',' -f1)
      echo "P3: $p3"
      echo "AVO: $avo"
      echo "Overall: $overall"
    fi
  else
    echo "Status: IDLE or COMPLETED"
    
    # Show all results
    echo ""
    echo "=== All Ultra-Aggressive Results ==="
    for log in ultra_aggressive_run*.log; do
      if [ -f "$log" ]; then
        echo "$(basename $log)"
      fi
    done
    
    if [ $completed -eq 5 ]; then
      echo ""
      echo "✓ All 5 experiments completed!"
      break
    fi
  fi
  
  echo ""
  echo "Press Ctrl+C to stop monitoring"
  sleep 30
done
