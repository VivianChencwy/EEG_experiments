#!/usr/bin/env python3
"""
Test script to verify the GPU fix works correctly
"""

import subprocess
import os
import time
from datetime import datetime

def test_gpu_subprocess():
    """Test that subprocess can access GPU correctly with conda environment"""
    print("=== Testing GPU access via subprocess ===")
    
    # Test GPU task (simulating our launch_process)
    env_proc = os.environ.copy()
    env_proc['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1
    
    result = subprocess.run(
        ['conda', 'run', '-n', 'eegtemp', 'python', '-c', '''
import torch
import os
print("CUDA_VISIBLE_DEVICES:", repr(os.environ.get("CUDA_VISIBLE_DEVICES")))
print("CUDA available:", torch.cuda.is_available())
print("Number of visible GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.randn(100, 100).to(device)
    print("GPU tensor created on device:", x.device)
    print("SUCCESS: GPU task working!")
else:
    print("ERROR: GPU not available!")
        '''],
        env=env_proc,
        capture_output=True,
        text=True
    )
    
    print("GPU subprocess output:")
    print(result.stdout)
    if result.stderr:
        print("GPU subprocess errors:")
        print(result.stderr)
    
    # Test CPU task (simulating our launch_process_cpu)
    env_proc_cpu = os.environ.copy()
    env_proc_cpu['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
    
    result_cpu = subprocess.run(
        ['conda', 'run', '-n', 'eegtemp', 'python', '-c', '''
import torch
import os
print("CUDA_VISIBLE_DEVICES:", repr(os.environ.get("CUDA_VISIBLE_DEVICES")))
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(100, 100).to(device)
print("Tensor created on device:", x.device)
print("SUCCESS: CPU task working!")
        '''],
        env=env_proc_cpu,
        capture_output=True,
        text=True
    )
    
    print("\nCPU subprocess output:")
    print(result_cpu.stdout)
    if result_cpu.stderr:
        print("CPU subprocess errors:")
        print(result_cpu.stderr)

if __name__ == "__main__":
    test_gpu_subprocess()
