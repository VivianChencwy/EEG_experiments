#!/usr/bin/env python3
"""
Run all EEG experiment configurations in parallel across GPUs 0-3.

This script DOES NOT change single-run behavior of main.py.
It uses environment variables supported by config.py to override settings per run.
"""

import os
import sys
import time
import shutil
import subprocess
from datetime import datetime

# Import dataset roots from config to reuse user paths
try:
    from config import P3_DATA_DIR, AVO_DATA_DIR
except Exception:
    P3_DATA_DIR, AVO_DATA_DIR = './P3 Raw Data BIDS-Compatible', './ds005863/ds005863'


def ensure_log_dir(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)


def build_env(base_env: dict, updates: dict) -> dict:
    env = base_env.copy()
    for k, v in updates.items():
        env[str(k)] = str(v)
    return env


def config_name(cfg: dict) -> str:
    mode = 'Combined' if cfg.get('USE_COMBINED_DATASETS') == '1' else cfg.get('DATASET')
    parts = [
        mode,
        f"el-{cfg.get('ELECTRODE_LIST')}",
        f"clf-{cfg.get('CLASSIFIER')}",
        f"sep-{cfg.get('SEPARATE_SUBJECT_CLASSIFICATION')}",
        f"usl-{cfg.get('USE_SUBJECT_LAYER')}",
    ]
    return '_'.join(parts).replace(' ', '')


def make_configs() -> list[dict]:
    configs: list[dict] = []

    electrodes = ['common', 'all']
    classifiers = ['ShallowFBCSPNet', 'lda']
    sep_opts = ['True', 'False']

    # Helper to add config
    def add_cfg(dataset: str, data_dir: str, use_combined: bool, electrode: str,
                classifier: str, separate_subject: bool, use_subject_layer: bool) -> None:
        cfg = {
            'DATASET': dataset,
            'DATA_DIR': data_dir,
            'USE_COMBINED_DATASETS': '1' if use_combined else '0',
            'ELECTRODE_LIST': electrode,
            'CLASSIFIER': classifier,
            'SEPARATE_SUBJECT_CLASSIFICATION': 'True' if separate_subject else 'False',
            'USE_SUBJECT_LAYER': 'True' if use_subject_layer else 'False',
            # Also pass dataset roots to be explicit
            'P3_DATA_DIR': P3_DATA_DIR,
            'AVO_DATA_DIR': AVO_DATA_DIR,
        }
        configs.append(cfg)

    # use_subject_layer = False: 18
    # P3 only: 8 (2 electrode × 2 classifier × 2 separate_subject)
    for electrode in electrodes:
        for classifier in classifiers:
            for sep in [True, False]:
                add_cfg(
                    dataset='P3 Raw Data BIDS-Compatible',
                    data_dir=P3_DATA_DIR,
                    use_combined=False,
                    electrode=electrode,
                    classifier=classifier,
                    separate_subject=sep,
                    use_subject_layer=False,
                )

    # ds005863 only: 8 (2 electrode × 2 classifier × 2 separate_subject)
    for electrode in electrodes:
        for classifier in classifiers:
            for sep in [True, False]:
                add_cfg(
                    dataset='ds005863',
                    data_dir=AVO_DATA_DIR,
                    use_combined=False,
                    electrode=electrode,
                    classifier=classifier,
                    separate_subject=sep,
                    use_subject_layer=False,
                )

    # Combined: 2 (1 electrode × 2 classifier × 1 separate_subject)
    for classifier in classifiers:
        add_cfg(
            dataset='use_combined_datasets',  # informational only
            data_dir=P3_DATA_DIR,  # unused in combined; still set
            use_combined=True,
            electrode='common',
            classifier=classifier,
            separate_subject=False,
            use_subject_layer=False,
        )

    # use_subject_layer = True: 5
    # P3 only + ShallowFBCSPNet + separate_subject_classification=False × 2 electrode
    for electrode in electrodes:
        add_cfg(
            dataset='P3 Raw Data BIDS-Compatible',
            data_dir=P3_DATA_DIR,
            use_combined=False,
            electrode=electrode,
            classifier='ShallowFBCSPNet',
            separate_subject=False,
            use_subject_layer=True,
        )

    # AVO only + ShallowFBCSPNet + separate_subject_classification=False × 2 electrode
    for electrode in electrodes:
        add_cfg(
            dataset='ds005863',
            data_dir=AVO_DATA_DIR,
            use_combined=False,
            electrode=electrode,
            classifier='ShallowFBCSPNet',
            separate_subject=False,
            use_subject_layer=True,
        )

    # Combined + ShallowFBCSPNet + separate_subject_classification=False + common
    add_cfg(
        dataset='use_combined_datasets',
        data_dir=P3_DATA_DIR,
        use_combined=True,
        electrode='common',
        classifier='ShallowFBCSPNet',
        separate_subject=False,
        use_subject_layer=True,
    )

    assert len(configs) == 23, f"Expected 23 configs, got {len(configs)}"
    return configs


def launch_process(env: dict, gpu_id: int) -> subprocess.Popen:
    env_proc = os.environ.copy()
    env_proc.update(env)
    env_proc['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Use environment-specific python to ensure correct environment with CUDA support
    proc = subprocess.Popen(
        ['/home/cwy/anaconda3/envs/eegtemp/bin/python', 'main.py'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env_proc,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    return proc


def launch_process_cpu(env: dict) -> subprocess.Popen:
    """Launch a CPU-only process (for lda classifier)."""
    env_proc = os.environ.copy()
    env_proc.update(env)
    # Set CUDA_VISIBLE_DEVICES to empty to force CPU usage
    env_proc['CUDA_VISIBLE_DEVICES'] = ''

    # Use environment-specific python to ensure correct environment
    proc = subprocess.Popen(
        ['/home/cwy/anaconda3/envs/eegtemp/bin/python', 'main.py'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env_proc,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    return proc


def main() -> int:
    # GPUs to use for neural network tasks
    gpu_ids = [0, 1, 2, 3]

    # Prepare configs
    configs = make_configs()
    
    # Separate GPU tasks (ShallowFBCSPNet) from CPU tasks (lda)
    gpu_configs = [cfg for cfg in configs if cfg.get('CLASSIFIER') == 'ShallowFBCSPNet']
    cpu_configs = [cfg for cfg in configs if cfg.get('CLASSIFIER') == 'lda']
    
    print(f"Total configs: {len(configs)}")
    print(f"GPU tasks (ShallowFBCSPNet): {len(gpu_configs)}")
    print(f"CPU tasks (lda): {len(cpu_configs)}")

    # Distribute GPU configs evenly across GPUs
    per_gpu_queues: dict[int, list[dict]] = {g: [] for g in gpu_ids}
    for i, cfg in enumerate(gpu_configs):
        per_gpu_queues[gpu_ids[i % len(gpu_ids)]].append(cfg)

    # Print distribution
    print("\nGPU task distribution:")
    for g in gpu_ids:
        print(f"  GPU {g}: {len(per_gpu_queues[g])} tasks")

    # Start ALL GPU tasks simultaneously
    running_gpu_processes = []  # List of (process, gpu_id, config_name) tuples
    print("\nStarting all GPU tasks simultaneously...")
    
    for gpu_id in gpu_ids:
        for config in per_gpu_queues[gpu_id]:
            proc = launch_process(config, gpu_id)
            cfg_name = config_name(config)
            running_gpu_processes.append((proc, gpu_id, cfg_name))
            print(f"[GPU {gpu_id}] Started: {cfg_name}")

    # Start CPU processes (can run multiple in parallel since they don't use GPU)
    max_cpu_parallel = 4  # Increase CPU parallelism since GPUs are fully utilized
    running_cpu_processes = []  # List of (process, config_name) tuples
    
    print("\nStarting CPU tasks...")
    for i, config in enumerate(cpu_configs[:max_cpu_parallel]):
        proc = launch_process_cpu(config)
        cfg_name = config_name(config)
        running_cpu_processes.append((proc, cfg_name))
        print(f"[CPU] Started: {cfg_name}")

    remaining_cpu_configs = cpu_configs[max_cpu_parallel:]
    remaining_cpu_index = 0

    # Print initial status
    print(f"\nRunning: {len(running_gpu_processes)} GPU tasks, {len(running_cpu_processes)} CPU tasks")
    print("All GPU tasks started! Waiting for completion...")
    
    # Monitor all processes
    while running_gpu_processes or running_cpu_processes:
        time.sleep(2)
        
        # Check GPU processes
        finished_gpu_indices = []
        for i, (proc, gpu_id, cfg_name) in enumerate(running_gpu_processes):
            ret = proc.poll()
            if ret is not None:
                finished_gpu_indices.append(i)
                print(f"[GPU {gpu_id}] Completed: {cfg_name}")
        
        # Remove finished GPU processes (reverse order to avoid index issues)
        for i in reversed(finished_gpu_indices):
            running_gpu_processes.pop(i)
        
        # Check CPU processes
        finished_cpu_indices = []
        for i, (proc, cfg_name) in enumerate(running_cpu_processes):
            ret = proc.poll()
            if ret is not None:
                finished_cpu_indices.append(i)
                print(f"[CPU] Completed: {cfg_name}")
        
        # Remove finished CPU processes and start new ones
        for i in reversed(finished_cpu_indices):
            running_cpu_processes.pop(i)
        
        # Start new CPU processes to maintain parallelism
        while (len(running_cpu_processes) < max_cpu_parallel and 
               remaining_cpu_index < len(remaining_cpu_configs)):
            config = remaining_cpu_configs[remaining_cpu_index]
            proc = launch_process_cpu(config)
            cfg_name = config_name(config)
            running_cpu_processes.append((proc, cfg_name))
            print(f"[CPU] Started: {cfg_name}")
            remaining_cpu_index += 1

    print("\n🎉 All experiments completed!")
    print(f"📊 Processed {len(gpu_configs)} GPU tasks and {len(cpu_configs)} CPU tasks")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


