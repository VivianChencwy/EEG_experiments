#!/bin/bash

# 顺序运行所有4个AVOsmall ablation实验
# 确保config.py中的配置是: P3=80, AVO=10 (AVO是小数据集)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="log_0909"

echo "=========================================="
echo "Running All Ablation Studies (AVOsmall)"
echo "Config: P3=80 trials, AVO=10 trials (AVO is target)"
echo "Start time: $(date)"
echo "=========================================="

# Ablation 1: Equal Weights
echo "[1/4] Running Ablation 1: Equal Weights..."
python main_ablation1_equal_weights_AVOsmall.py > ${LOG_DIR}/Ablation1_AVOsmall_${TIMESTAMP}.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Ablation 1 completed successfully"
else
    echo "✗ Ablation 1 failed"
    exit 1
fi

# Ablation 2: Fixed Weights
echo "[2/4] Running Ablation 2: Fixed Weights..."
python main_ablation2_fixed_weights_AVOsmall.py > ${LOG_DIR}/Ablation2_AVOsmall_${TIMESTAMP}.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Ablation 2 completed successfully"
else
    echo "✗ Ablation 2 failed"
    exit 1
fi

# Ablation 3: No MMD
echo "[3/4] Running Ablation 3: No MMD..."
python main_ablation3_no_mmd_AVOsmall.py > ${LOG_DIR}/Ablation3_AVOsmall_${TIMESTAMP}.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Ablation 3 completed successfully"
else
    echo "✗ Ablation 3 failed"
    exit 1
fi

# Ablation 4: No Split BN
echo "[4/4] Running Ablation 4: No Split BN..."
python main_ablation4_no_split_bn_AVOsmall.py > ${LOG_DIR}/Ablation4_AVOsmall_${TIMESTAMP}.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Ablation 4 completed successfully"
else
    echo "✗ Ablation 4 failed"
    exit 1
fi

echo "=========================================="
echo "All ablation studies completed!"
echo "End time: $(date)"
echo "=========================================="

# 生成结果摘要
echo ""
echo "Results summary:"
ls -lh ablation_results_AVOsmall/*.csv | tail -8
