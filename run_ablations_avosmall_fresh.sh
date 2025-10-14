#!/bin/bash

# 确认配置
echo "================================================"
echo "Running ALL Ablation Studies for AVO as Target"
echo "Configuration: P3=80, AVO=10 (AVO is small/target)"
echo "================================================"
echo ""

# 验证config.py配置
echo "Checking config.py..."
P3_TRIALS=$(grep "NESTED_CV_TRIALS_PER_SUBJECT_P3 =" config.py | grep -v "#" | awk '{print $3}')
AVO_TRIALS=$(grep "NESTED_CV_TRIALS_PER_SUBJECT_AVO =" config.py | grep -v "#" | awk '{print $3}')
echo "  P3 trials per subject: $P3_TRIALS"
echo "  AVO trials per subject: $AVO_TRIALS"

if [ "$P3_TRIALS" != "80" ] || [ "$AVO_TRIALS" != "10" ]; then
    echo "ERROR: Config incorrect! Should be P3=80, AVO=10"
    exit 1
fi

echo "✓ Configuration correct"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 运行4个ablation实验
for i in 1 2 3 4; do
    case $i in
        1) NAME="Equal Weights"; SCRIPT="main_ablation1_equal_weights_AVOsmall.py" ;;
        2) NAME="Fixed Weights"; SCRIPT="main_ablation2_fixed_weights_AVOsmall.py" ;;
        3) NAME="No MMD"; SCRIPT="main_ablation3_no_mmd_AVOsmall.py" ;;
        4) NAME="No Split BN"; SCRIPT="main_ablation4_no_split_bn_AVOsmall.py" ;;
    esac

    echo "[$i/4] Running Ablation $i: $NAME"
    echo "  Script: $SCRIPT"
    echo "  Start time: $(date)"

    python $SCRIPT > log_0909/Ablation${i}_AVOsmall_${TIMESTAMP}.log 2>&1

    if [ $? -eq 0 ]; then
        echo "  ✓ Completed successfully at $(date)"
        # 显示生成的CSV文件
        ls -lh ablation_results_AVOsmall/ablation${i}_*.csv 2>/dev/null | tail -2
    else
        echo "  ✗ FAILED"
        exit 1
    fi
    echo ""
done

echo "================================================"
echo "All Ablation Studies Completed!"
echo "End time: $(date)"
echo "================================================"
echo ""
echo "Generated files:"
ls -lht ablation_results_AVOsmall/*.csv | head -10
