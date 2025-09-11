@echo off
echo ========================================
echo 启动改进EEG分类实验
echo ========================================
echo.

echo 激活conda环境...
call conda activate eeg_realtime
if %ERRORLEVEL% NEQ 0 (
    echo 错误：无法激活eeg_realtime环境
    echo 尝试激活base环境...
    call conda activate base
)

echo.
echo 检查Python环境:
python -c "import sys; print('Python:', sys.executable)"

echo.
echo 检查必要的库:
python -c "
try:
    import mne
    import numpy as np
    import torch
    import sklearn
    from mne.preprocessing import ICA
    print('✓ 所有必要的库都已安装')
    print('✓ ICA可用 - 将启用伪影去除')
except ImportError as e:
    print('⚠️  缺少库:', e)
    print('   部分功能可能不可用')
"

echo.
echo ========================================
echo 开始运行改进实验...
echo ========================================
echo.
echo 预期改进:
echo   • 当前baseline: 71.3%%
echo   • 预期准确率: 78-85%%
echo   • 主要改进: ICA去伪影 + ERP优化频率
echo.

python run_improved_experiment.py

echo.
echo ========================================
echo 实验完成！
echo ========================================
echo.
echo 查看结果日志:
echo   dir log_*\*Improved*
echo.
pause