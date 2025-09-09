@echo off
echo 运行P3数据集实验...
echo.

REM 激活eeg_realtime环境
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
    exit(1)
"
echo.

REM 使用P3配置
copy config_p3.py config.py
echo ✓ 使用P3数据集配置

echo 开始运行P3 EEG实验...
python main.py

echo.
echo P3实验完成！
pause