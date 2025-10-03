#!/bin/bash
# Master overnight optimization script
# Automatically runs both AVO and P3 scenarios

LOG_FILE="master_optimization_$(date +%Y%m%d_%H%M%S).log"

{
  echo "=== MASTER OVERNIGHT OPTIMIZATION ==="
  echo "Start time: $(date)"
  echo ""
  
  # Wait for current AVO experiments to complete
  echo "Waiting for AVO scenario experiments to complete..."
  while ps aux | grep -q "[p]ython main_tfdwt.py"; do
    sleep 60
  done
  
  echo "AVO scenario completed at: $(date)"
  echo ""
  
  # Analyze AVO results
  echo "=== AVO Scenario Results ==="
  for log in ultra_aggressive_run*.log; do
    if [ -f "$log" ]; then
      echo "Log: $(basename $log)"
    fi
  done
  echo ""
  
  # Run P3 scenario
  echo "=== Starting P3 Scenario ==="
  echo "Start time: $(date)"
  /home/vivian/eeg/EEG_experiments/run_p3_scenario.sh
  
  echo ""
  echo "P3 scenario completed at: $(date)"
  echo ""
  
  # Generate final report
  echo "=== Generating Final Report ==="
  python3 /home/vivian/eeg/EEG_experiments/generate_final_report.py
  
  echo ""
  echo "=== MASTER OPTIMIZATION COMPLETE ==="
  echo "End time: $(date)"
  
} 2>&1 | tee "$LOG_FILE"
