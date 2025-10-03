#!/bin/bash
while true; do
  clear
  echo "=== 极限优化进度监控 ==="
  echo "时间: $(date)"
  echo ""
  
  # 显示完整日志
  if [ -f extreme_avo_master.log ]; then
    echo "=== 实验结果 ==="
    cat extreme_avo_master.log
    echo ""
  fi
  
  # 检查进程
  if ps aux | grep -q "[p]ython main_tfdwt.py"; then
    echo "状态: 运行中..."
  else
    echo "状态: 已完成"
    
    # 统计结果
    echo ""
    echo "=== 结果统计 ==="
    grep "AVO Result:" extreme_avo_master.log | while read line; do
      avo=$(echo $line | awk '{print $3}')
      if (( $(echo "$avo >= 0.66" | bc -l) )); then
        echo "✓ $avo"
      else
        echo "✗ $avo"
      fi
    done
    
    break
  fi
  
  sleep 60
done
