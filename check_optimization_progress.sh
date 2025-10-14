#!/bin/bash
# Monitor optimization progress

LOG_FILE="optimized_run1.log"

echo "=== Optimization Progress Monitor ==="
echo "Checking log file: $LOG_FILE"
echo ""

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found yet"
    exit 1
fi

# Get file size
SIZE=$(wc -c < "$LOG_FILE")
echo "Log file size: $SIZE bytes"
echo ""

# Check if process is running
PID=$(ps aux | grep "python main_tfdwt.py" | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    echo "Process running: PID $PID"
else
    echo "Process completed or not running"
fi
echo ""

# Show last lines
echo "=== Last 40 lines of log ==="
tail -40 "$LOG_FILE"
