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


def launch_process(env: dict, gpu_id: int, log_dir: str, idx: int) -> subprocess.Popen:
    env_proc = os.environ.copy()
    env_proc.update(env)
    env_proc['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    cfg_tag = config_name(env)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f"{idx:02d}_{cfg_tag}_{timestamp}.log")

    with open(log_path, 'wb') as log_file:
        proc = subprocess.Popen(
            [sys.executable, 'main.py'],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env_proc,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

    return proc


def launch_process_cpu(env: dict, log_dir: str, idx: int) -> subprocess.Popen:
    """Launch a CPU-only process (for lda classifier)."""
    env_proc = os.environ.copy()
    env_proc.update(env)
    # Set CUDA_VISIBLE_DEVICES to empty to force CPU usage
    env_proc['CUDA_VISIBLE_DEVICES'] = ''

    cfg_tag = config_name(env)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f"{idx:02d}_{cfg_tag}_{timestamp}_CPU.log")

    with open(log_path, 'wb') as log_file:
        proc = subprocess.Popen(
            [sys.executable, 'main.py'],
            stdout=log_file,
            stderr=subprocess.STDOUT,
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

    log_dir = os.path.join('log_', 'parallel')
    ensure_log_dir(log_dir)

    # Track running processes
    running_gpu: dict[int, tuple[subprocess.Popen, int]] = {}  # GPU processes
    running_cpu: list[tuple[subprocess.Popen, int]] = []  # CPU processes
    next_gpu_index: dict[int, int] = {g: 0 for g in gpu_ids}
    next_cpu_index = 0
    cfg_global_index = 0

    # Start one GPU process per GPU if available
    for g in gpu_ids:
        if next_gpu_index[g] < len(per_gpu_queues[g]):
            proc = launch_process(per_gpu_queues[g][next_gpu_index[g]], g, log_dir, cfg_global_index)
            running_gpu[g] = (proc, cfg_global_index)
            next_gpu_index[g] += 1
            cfg_global_index += 1

    # Start CPU processes (can run multiple in parallel since they don't use GPU)
    max_cpu_parallel = 2  # Adjust based on your system
    while len(running_cpu) < max_cpu_parallel and next_cpu_index < len(cpu_configs):
        proc = launch_process_cpu(cpu_configs[next_cpu_index], log_dir, cfg_global_index)
        running_cpu.append((proc, cfg_global_index))
        next_cpu_index += 1
        cfg_global_index += 1

    # Loop until all queues are drained
    while running_gpu or running_cpu:
        time.sleep(2)
        
        # Check GPU processes
        finished_gpus = []
        for g, (proc, _) in running_gpu.items():
            ret = proc.poll()
            if ret is not None:
                finished_gpus.append(g)
        
        for g in finished_gpus:
            # Start next GPU task on this GPU if available
            if next_gpu_index[g] < len(per_gpu_queues[g]):
                proc = launch_process(per_gpu_queues[g][next_gpu_index[g]], g, log_dir, cfg_global_index)
                running_gpu[g] = (proc, cfg_global_index)
                next_gpu_index[g] += 1
                cfg_global_index += 1
            else:
                # No more jobs for this GPU
                del running_gpu[g]
        
        # Check CPU processes
        finished_cpu_indices = []
        for i, (proc, _) in enumerate(running_cpu):
            ret = proc.poll()
            if ret is not None:
                finished_cpu_indices.append(i)
        
        # Remove finished CPU processes and start new ones
        for i in reversed(finished_cpu_indices):  # Reverse to avoid index issues
            running_cpu.pop(i)
        
        # Start new CPU processes to maintain parallelism
        while len(running_cpu) < max_cpu_parallel and next_cpu_index < len(cpu_configs):
            proc = launch_process_cpu(cpu_configs[next_cpu_index], log_dir, cfg_global_index)
            running_cpu.append((proc, cfg_global_index))
            next_cpu_index += 1
            cfg_global_index += 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())


