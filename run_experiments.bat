@echo off
echo 激活conda环境并运行EEG实验...
echo.

REM 激活eeg_realtime环境并运行主脚本
call conda activate eeg_realtime
if %ERRORLEVEL% NEQ 0 (
    echo 错误：无法激活eeg_realtime环境
    echo 尝试激活base环境...
    call conda activate base
)

echo 当前Python环境:
python -c "import sys; print('Python:', sys.executable)"
echo.

echo 检查必要的库:
python -c "
try:
    import mne
    import numpy as np
    import torch
    import sklearn
    print('✓ 所有必要的库都已安装')
except ImportError as e:
    print('✗ 缺少库:', e)
"
echo.

echo 开始运行EEG实验...
python main.py

echo.
echo 实验完成！
pause