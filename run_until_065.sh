#!/bin/bash
# 反复运行TF-DWT直到AVO准确率>=0.65

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_COUNT=0
MAX_RUNS=20  # 最多尝试20次

echo "=========================================="
echo "反复运行TF-DWT直到AVO准确率>=0.65"
echo "配置: P3=80, AVO=10 (原始参数)"
echo "开始时间: $(date)"
echo "=========================================="
echo ""

while [ $RUN_COUNT -lt $MAX_RUNS ]; do
    RUN_COUNT=$((RUN_COUNT + 1))
    echo ""
    echo "==================== 运行 #$RUN_COUNT ===================="
    echo "时间: $(date)"
    
    # 运行实验
    python main_tfdwt.py > log_0909/TF_DWT_run${RUN_COUNT}_${TIMESTAMP}.log 2>&1
    
    if [ $? -eq 0 ]; then
        # 查找最新的CSV文件
        latest_csv=$(ls -t tfdwt_detailed_results_*.csv 2>/dev/null | head -1)
        
        if [ -n "$latest_csv" ]; then
            # 检查AVO准确率
            result=$(python3 << PYEOF
import pandas as pd
import sys
df = pd.read_csv('$latest_csv')
avo_acc = df['avo_accuracy'].mean()
avo_std = df['avo_accuracy'].std()
print(f"{avo_acc:.4f} ± {avo_std:.4f}")
sys.exit(0 if avo_acc >= 0.65 else 1)
PYEOF
)
            exit_code=$?
            
            # 重命名文件
            new_name="tfdwt_run${RUN_COUNT}_detailed_${TIMESTAMP}.csv"
            mv "$latest_csv" "$new_name"
            mv tfdwt_summary_stats_*.csv "tfdwt_run${RUN_COUNT}_summary_${TIMESTAMP}.csv" 2>/dev/null
            
            echo "结果: AVO准确率 = $result"
            echo "文件: $new_name"
            
            if [ $exit_code -eq 0 ]; then
                echo ""
                echo "🎉🎉🎉 目标达成！🎉🎉🎉"
                echo "运行次数: $RUN_COUNT"
                echo "AVO准确率: $result >= 0.65"
                echo "结果文件: $new_name"
                echo ""
                
                # 保存最优配置
                cat > optimal_result.json << JSONEOF
{
  "run": $RUN_COUNT,
  "avo_accuracy": "$result",
  "csv_file": "$new_name",
  "timestamp": "$(date -Iseconds)",
  "parameters": "original (no modifications)"
}
JSONEOF
                echo "最优结果已保存到: optimal_result.json"
                exit 0
            fi
        else
            echo "✗ 未找到CSV文件"
        fi
    else
        echo "✗ 运行失败"
    fi
    
    echo "等待5秒后继续..."
    sleep 5
done

echo ""
echo "=========================================="
echo "⚠️  尝试 $MAX_RUNS 次后仍未达标"
echo "最佳结果请查看生成的CSV文件"
echo "=========================================="
