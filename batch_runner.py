"""
Batch runner for multiple configurations.

Runs:
- main.py: 4 configurations
- main_tfdwt.py: 2 configurations

It does NOT modify the project's original config.py. For each run, it creates
an isolated temporary directory containing a tailored config.py that overrides
only the requested fields, and prepends that directory to PYTHONPATH so that
imports resolve to the temporary config first. All other modules are imported
from the project as usual. Output is saved to individual log files in a batch
log directory, maintaining the same format as direct script execution.

Usage:
  python batch_runner.py
"""

import os
import re
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent
BASE_CONFIG_PATH = PROJECT_ROOT / 'config.py'

# Create batch log directory
BATCH_LOG_DIR = PROJECT_ROOT / 'log_batch'
BATCH_LOG_DIR.mkdir(exist_ok=True)


def read_base_config() -> str:
    return BASE_CONFIG_PATH.read_text(encoding='utf-8')


def apply_overrides(config_text: str, overrides: dict) -> str:
    """Return a new config.py text with specific top-level assignments overridden.

    Supported keys: use_combined_datasets (bool), data_dir (str expr), dataset (str),
                    NESTED_CV_TRIALS_PER_SUBJECT_P3 (int), NESTED_CV_TRIALS_PER_SUBJECT_AVO (int)
    """
    lines = config_text.splitlines()

    def set_line(prefix: str, value_src: str):
        nonlocal lines
        pat = re.compile(rf"^({re.escape(prefix)}\s*=).*$")
        replaced = False
        for i, line in enumerate(lines):
            if pat.match(line.strip()):
                lines[i] = f"{prefix} = {value_src}"
                replaced = True
                break
        if not replaced:
            # Append if not found (shouldn't happen with current config layout)
            lines.append(f"{prefix} = {value_src}")

    if 'use_combined_datasets' in overrides:
        set_line('use_combined_datasets', 'True' if overrides['use_combined_datasets'] else 'False')
    if 'data_dir' in overrides:
        # data_dir expects a symbol like P3_DATA_DIR or AVO_DATA_DIR
        set_line('data_dir', overrides['data_dir'])
    if 'dataset' in overrides:
        set_line('dataset', repr(overrides['dataset']))
    if 'NESTED_CV_TRIALS_PER_SUBJECT_P3' in overrides:
        set_line('NESTED_CV_TRIALS_PER_SUBJECT_P3', str(overrides['NESTED_CV_TRIALS_PER_SUBJECT_P3']))
    if 'NESTED_CV_TRIALS_PER_SUBJECT_AVO' in overrides:
        set_line('NESTED_CV_TRIALS_PER_SUBJECT_AVO', str(overrides['NESTED_CV_TRIALS_PER_SUBJECT_AVO']))

    return "\n".join(lines) + "\n"


def run_case(case_name: str, target_script: str, overrides: dict) -> int:
    base = read_base_config()
    mod = apply_overrides(base, overrides)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"batch_cfg_{case_name}_"))
    
    # Create log file for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{case_name}_{timestamp}.log"
    log_path = BATCH_LOG_DIR / log_filename
    
    try:
        (tmp_dir / 'config.py').write_text(mod, encoding='utf-8')
        
        # Debug: Print the generated config to verify overrides
        print(f"\n===== Running {case_name} =====")
        print(f"Overrides: {overrides}")
        print(f"Log file: {log_path}")
        print(f"Temp config dir: {tmp_dir}")
        
        # Verify the generated config contains correct values
        print("Generated config preview:")
        config_lines = mod.splitlines()
        for line in config_lines:
            if any(key in line for key in ['NESTED_CV_TRIALS_PER_SUBJECT', 'use_combined_datasets', 'data_dir', 'dataset']):
                print(f"  {line.strip()}")
        
        env = os.environ.copy()
        # Prepend tmp dir so 'import config' uses the overridden one
        py_path = f"{str(tmp_dir)}:{str(PROJECT_ROOT)}"
        env['PYTHONPATH'] = py_path if 'PYTHONPATH' not in env else f"{str(tmp_dir)}:{env['PYTHONPATH']}"
        
        # Also set PYTHONPATH as an environment variable to ensure it's used
        # Use -c to modify sys.path at runtime to ensure temp dir is first
        # Keep working directory as project root for data path resolution
        python_code = f"""
import sys
import os
sys.path.insert(0, '{str(tmp_dir)}')
# Set environment variable to point to temp config
os.environ['CONFIG_OVERRIDE_PATH'] = '{str(tmp_dir / "config.py")}'
import runpy
runpy.run_path('{str(PROJECT_ROOT / target_script)}', run_name='__main__')
"""
        cmd = [sys.executable, '-c', python_code]
        
        # Run with both console output and log file
        print(f"\n[{case_name}] Starting experiment...")
        with open(log_path, 'w', encoding='utf-8') as log_file:
            proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, 
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  universal_newlines=True, bufsize=1)
            
            # Read output line by line and write to both console and log
            while True:
                output = proc.stdout.readline()
                if output == '' and proc.poll() is not None:
                    break
                if output:
                    # Write to log file
                    log_file.write(output)
                    log_file.flush()
                    
                    # Print key progress lines to console
                    line = output.strip()
                    if any(keyword in line.lower() for keyword in [
                        'fold', 'epoch', 'accuracy', 'loss', 'completed', 'error', 'failed',
                        'processing', 'loading', 'training', 'testing', 'validation',
                        'p3 dataset:', 'avo dataset:', 'nested cv mode'
                    ]):
                        print(f"[{case_name}] {line}")
            
            proc.wait()
        
        # Print completion status
        print(f"\n[{case_name}] Completed with return code {proc.returncode}")
        if proc.returncode == 0:
            print(f"[{case_name}] ✓ SUCCESS - Log saved to: {log_path}")
        else:
            print(f"[{case_name}] ✗ FAILED - Check log: {log_path}")
            
        return proc.returncode
    finally:
        # Clean temp config directory
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    print(f"Batch runner starting - logs will be saved to: {BATCH_LOG_DIR}")
    print(f"Batch run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    cases = [
        # main.py (4 runs)
        (
            'main_P3_20',
            'main.py',
            {
                'data_dir': 'P3_DATA_DIR',
                'dataset': 'P3 Raw Data BIDS-Compatible',
                'use_combined_datasets': False,
                'NESTED_CV_TRIALS_PER_SUBJECT_P3': 20,
                'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 20,  # Keep consistent, but only P3 will be used
            },
        ),
        (
            'main_AVO_20',
            'main.py',
            {
                'data_dir': 'AVO_DATA_DIR',
                'dataset': 'ds005863',
                'use_combined_datasets': False,
                'NESTED_CV_TRIALS_PER_SUBJECT_P3': 0,  # Keep consistent, but only AVO will be used
                'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 20,
            },
        ),
        # (
        #     'main_combined_P3_200_AVO_20',
        #     'main.py',
        #     {
        #         'use_combined_datasets': True,
        #         'data_dir': 'P3_DATA_DIR',
        #         'dataset': 'use_combined_datasets',
        #         'NESTED_CV_TRIALS_PER_SUBJECT_P3': 200,
        #         'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 20,
        #     },
        # ),
        # (
        #     'main_combined_P3_20_AVO_200',
        #     'main.py',
        #     {
        #         'use_combined_datasets': True,
        #         'data_dir': 'P3_DATA_DIR',
        #         'dataset': 'use_combined_datasets',
        #         'NESTED_CV_TRIALS_PER_SUBJECT_P3': 20,
        #         'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 200,
        #     },
        # ),
        # # main_tfdwt.py (2 runs)
        # (
        #     'tfdwt_combined_P3_200_AVO_20',
        #     'main_tfdwt.py',
        #     {
        #         'use_combined_datasets': True,
        #         'data_dir': 'P3_DATA_DIR',
        #         'dataset': 'use_combined_datasets',
        #         'NESTED_CV_TRIALS_PER_SUBJECT_P3': 200,
        #         'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 20,
        #     },
        # ),
        # (
        #     'tfdwt_combined_P3_20_AVO_200',
        #     'main_tfdwt.py',
        #     {
        #         'use_combined_datasets': True,
        #         'data_dir': 'P3_DATA_DIR',
        #         'dataset': 'use_combined_datasets',
        #         'NESTED_CV_TRIALS_PER_SUBJECT_P3': 20,
        #         'NESTED_CV_TRIALS_PER_SUBJECT_AVO': 200,
        #     },
        # ),
    ]

    failures = 0
    successful_logs = []
    
    for i, (name, script, ov) in enumerate(cases, 1):
        print(f"\n{'='*60}")
        print(f"Running case {i}/{len(cases)}: {name}")
        print(f"{'='*60}")
        
        rc = run_case(name, script, ov)
        if rc != 0:
            print(f"[FAILED] {name} returned code {rc}")
            failures += 1
        else:
            print(f"[SUCCESS] {name}")
            successful_logs.append(name)

    print(f"\n{'='*60}")
    print("BATCH RUN SUMMARY")
    print(f"{'='*60}")
    print(f"Total cases: {len(cases)}")
    print(f"Successful: {len(successful_logs)}")
    print(f"Failed: {failures}")
    print(f"Log directory: {BATCH_LOG_DIR}")
    
    if successful_logs:
        print(f"\nSuccessful runs:")
        for name in successful_logs:
            print(f"  - {name}")
    
    if failures:
        print(f"\nFailed runs:")
        for name, script, ov in cases:
            if name not in successful_logs:
                print(f"  - {name}")
        print(f"\nCompleted with {failures} failures out of {len(cases)} total cases")
        print("Note: Some experiments failed, but batch runner completed all cases.")
    else:
        print("\nAll runs completed successfully")


if __name__ == '__main__':
    main()


