#!/usr/bin/env python3
"""
Run all EEG experiment configurations sequentially.

This script DOES NOT change single-run behavior of main.py.
It uses environment variables supported by config.py to override settings per run.
Each configuration is run one by one, equivalent to running main.py with each configuration.
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
    P3_DATA_DIR, AVO_DATA_DIR = '../P3_Raw_Data _BIDS-Compatible', '../ds005863/ds005863'


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
    classifiers = ['ShallowFBCSPNet']  # Only using ShallowFBCSPNet, removed lda
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

    # use_subject_layer = False: 9
    # P3 only: 4 (2 electrode × 1 classifier × 2 separate_subject)
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

    # ds005863 only: 4 (2 electrode × 1 classifier × 2 separate_subject)
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

    # Combined: 1 (1 electrode × 1 classifier × 1 separate_subject)
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

    # use_subject_layer = True: 5 (unchanged since these were already ShallowFBCSPNet only)
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

    assert len(configs) == 14, f"Expected 14 configs, got {len(configs)}"
    return configs


def run_single_experiment(env: dict) -> int:
    """Run a single experiment configuration and wait for completion."""
    env_proc = os.environ.copy()
    env_proc.update(env)
    
    # All experiments now use GPU since we only have ShallowFBCSPNet
    env_proc['CUDA_VISIBLE_DEVICES'] = '0'

    # Use environment-specific python to ensure correct environment
    proc = subprocess.Popen(
        ['/home/cwy/anaconda3/envs/eegtemp/bin/python', 'main.py'],
        env=env_proc,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    # Wait for process to complete
    return_code = proc.wait()
    return return_code


def main() -> int:
    # Prepare configs
    configs = make_configs()
    
    print(f"Total configs: {len(configs)}")
    print(f"Running all experiments sequentially...")
    print(f"Each experiment will run exactly like main.py with the specific configuration\n")

    start_time = time.time()
    successful_runs = 0
    failed_runs = 0
    
    # Run each configuration sequentially
    for i, config in enumerate(configs, 1):
        cfg_name = config_name(config)
        # All experiments now use GPU since we only have ShallowFBCSPNet
        
        print(f"[{i}/{len(configs)}] Running: {cfg_name} (GPU)")
        
        experiment_start = time.time()
        return_code = run_single_experiment(config)
        experiment_time = time.time() - experiment_start
        
        if return_code == 0:
            successful_runs += 1
            print(f"[{i}/{len(configs)}] Completed: {cfg_name} (took {experiment_time:.1f}s)")
        else:
            failed_runs += 1
            print(f"[{i}/{len(configs)}] Failed: {cfg_name} (return code: {return_code})")
        
        print()  # Empty line for readability

    total_time = time.time() - start_time
    
    print(" All experiments completed!")
    print(f" Summary:")
    print(f"   Total experiments: {len(configs)}")
    print(f"   Successful: {successful_runs}")
    print(f"   Failed: {failed_runs}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    
    return 0 if failed_runs == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())


