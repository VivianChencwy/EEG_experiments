#!/bin/bash

LOG_FILE="master_extreme_$(date +%Y%m%d_%H%M%S).log"

{
  echo "=== EXTREME OPTIMIZATION MASTER CONTROLLER ==="
  echo "Start: $(date)"
  echo ""
  
  # Wait for current AVO experiment to complete
  echo "Waiting for ongoing AVO experiments..."
  while ps aux | grep -q "[p]ython main_tfdwt.py"; do
    sleep 60
  done
  
  echo "Previous experiments completed at: $(date)"
  echo ""
  
  # Run P3 scenario
  echo "=== Starting P3 Extreme Scenario ==="
  /home/vivian/eeg/EEG_experiments/run_extreme_p3.sh
  
  echo ""
  echo "P3 scenario completed at: $(date)"
  
  # Generate final report
  echo ""
  echo "=== Generating Final Report ==="
  python3 /home/vivian/eeg/EEG_experiments/generate_final_report.py
  
  echo ""
  echo "=== EXTREME OPTIMIZATION COMPLETE ==="
  echo "End: $(date)"
  
} 2>&1 | tee "$LOG_FILE"
