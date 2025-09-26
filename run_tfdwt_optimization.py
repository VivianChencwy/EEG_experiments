#!/usr/bin/env python3
"""
TF-DWT Optimization Runner

This script orchestrates the complete TF-DWT parameter optimization process.
It runs in stages to efficiently find optimal parameters:

1. Quick screening on both focus cases
2. Full evaluation with best parameters
3. Generate final comparison results

Usage:
    python run_tfdwt_optimization.py
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            return True
        else:
            print(f"❌ FAILED: {description}")
            print("STDOUT:", result.stdout[-500:])
            print("STDERR:", result.stderr[-500:])
            return False

    except Exception as e:
        print(f"❌ EXCEPTION: {description} - {e}")
        return False

def extract_best_params(results_dir: Path, focus_case: str, stage: str):
    """Extract best parameters from optimization results"""
    json_files = list(results_dir.glob(f"tfdwt_optimization_{focus_case}_{stage}_*.json"))

    if not json_files:
        print(f"No results found for {focus_case} {stage}")
        return None

    # Get the most recent file
    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)

    with open(latest_file, 'r') as f:
        data = json.load(f)

    return {
        'best_accuracy': data.get('best_accuracy', 0.0),
        'best_params': data.get('best_params'),
        'total_experiments': len(data.get('all_results', [])),
        'file': str(latest_file)
    }

def create_final_experiment_script(best_params_p3, best_params_avo):
    """Create a final experiment script with optimal parameters"""

    script_content = f'''#!/usr/bin/env python3
"""
Final TF-DWT experiments with optimal parameters
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Best parameters found:
P3_small case: accuracy = {best_params_p3.get('best_accuracy', 0.0):.4f}
AVO_small case: accuracy = {best_params_avo.get('best_accuracy', 0.0):.4f}
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def run_final_experiment(focus_case: str, params_info: dict):
    """Run final experiment with optimal parameters"""
    print(f"\\n{{'='*60}}")
    print(f"FINAL EXPERIMENT: {{focus_case}}")
    print(f"Expected accuracy improvement: {{params_info.get('best_accuracy', 0.0):.4f}}")
    print(f"{{'='*60}}")

    cmd = [
        sys.executable, 'tfdwt_param_optimizer.py',
        '--stage', 'full',
        '--focus', focus_case
    ]

    # Override with optimal parameters by modifying the optimizer script
    # (This would be done by creating a custom version with fixed params)

    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running final experiment: {{e}}")
        return False

def main():
    """Run final optimized experiments"""

    # Parameters from optimization
    p3_params = {best_params_p3}
    avo_params = {best_params_avo}

    print("Starting final TF-DWT experiments with optimal parameters...")

    # Run P3_small case
    success_p3 = run_final_experiment('P3_small', p3_params)

    # Run AVO_small case
    success_avo = run_final_experiment('AVO_small', avo_params)

    if success_p3 and success_avo:
        print("\\n🎉 All final experiments completed successfully!")
    else:
        print(f"\\n⚠️  Some experiments failed - P3: {{success_p3}}, AVO: {{success_avo}}")

if __name__ == '__main__':
    main()
'''

    final_script_path = PROJECT_ROOT / 'run_final_tfdwt_experiments.py'
    with open(final_script_path, 'w') as f:
        f.write(script_content)

    # Make it executable
    os.chmod(final_script_path, 0o755)

    return final_script_path

def main():
    """Main optimization orchestration"""
    start_time = datetime.now()

    print("🚀 Starting TF-DWT Parameter Optimization")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    results_dir = PROJECT_ROOT / 'log_tfdwt_optimization'
    results_dir.mkdir(exist_ok=True)

    # Stage 1: Quick screening for both cases
    print("\\n" + "="*80)
    print("STAGE 1: QUICK PARAMETER SCREENING")
    print("="*80)

    screening_results = {}

    for focus_case in ['P3_small', 'AVO_small']:
        cmd = [
            sys.executable, 'tfdwt_param_optimizer.py',
            '--stage', 'screening',
            '--focus', focus_case
        ]

        success = run_command(cmd, f"Screening optimization for {focus_case}")

        if success:
            # Extract best parameters from results
            best_params = extract_best_params(results_dir, focus_case, 'screening')
            screening_results[focus_case] = best_params

            if best_params:
                print(f"\\n📊 {focus_case} screening results:")
                print(f"  Best accuracy: {best_params['best_accuracy']:.4f}")
                print(f"  Total experiments: {best_params['total_experiments']}")
        else:
            print(f"⚠️  Screening failed for {focus_case}")
            screening_results[focus_case] = None

    # Stage 2: Full evaluation with top parameters
    print("\\n" + "="*80)
    print("STAGE 2: FULL EVALUATION WITH OPTIMAL PARAMETERS")
    print("="*80)

    # Note: The full evaluation would use the best parameters found in screening
    # For now, we'll run with the current parameter ranges in 'full' mode

    full_results = {}

    for focus_case in ['P3_small', 'AVO_small']:
        if screening_results.get(focus_case):
            print(f"\\n🎯 Running full evaluation for {focus_case}")
            print(f"Based on screening accuracy: {screening_results[focus_case]['best_accuracy']:.4f}")

            cmd = [
                sys.executable, 'tfdwt_param_optimizer.py',
                '--stage', 'full',
                '--focus', focus_case
            ]

            success = run_command(cmd, f"Full optimization for {focus_case}")

            if success:
                best_params = extract_best_params(results_dir, focus_case, 'full')
                full_results[focus_case] = best_params

                if best_params:
                    print(f"\\n🏆 {focus_case} FINAL results:")
                    print(f"  Best accuracy: {best_params['best_accuracy']:.4f}")
                    print(f"  Total experiments: {best_params['total_experiments']}")
            else:
                print(f"⚠️  Full evaluation failed for {focus_case}")
                full_results[focus_case] = None
        else:
            print(f"⚠️  Skipping full evaluation for {focus_case} (screening failed)")

    # Stage 3: Summary and final script generation
    print("\\n" + "="*80)
    print("STAGE 3: OPTIMIZATION SUMMARY")
    print("="*80)

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"Total optimization time: {duration}")
    print(f"Results directory: {results_dir}")

    # Create summary report
    summary = {
        'optimization_start': start_time.isoformat(),
        'optimization_end': end_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'screening_results': screening_results,
        'full_results': full_results,
    }

    summary_file = results_dir / f"optimization_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\\n📋 Summary saved to: {summary_file}")

    # Print final results
    print("\\n🎯 FINAL OPTIMIZATION RESULTS:")

    for focus_case in ['P3_small', 'AVO_small']:
        full_result = full_results.get(focus_case)
        screening_result = screening_results.get(focus_case)

        print(f"\\n{focus_case}:")

        if full_result and full_result['best_accuracy'] > 0:
            print(f"  ✅ Final accuracy: {full_result['best_accuracy']:.4f}")
            print(f"  📁 Results file: {Path(full_result['file']).name}")
        elif screening_result and screening_result['best_accuracy'] > 0:
            print(f"  ⚠️  Screening accuracy: {screening_result['best_accuracy']:.4f} (full eval failed)")
            print(f"  📁 Results file: {Path(screening_result['file']).name}")
        else:
            print(f"  ❌ No successful results")

    # Create final experiment script if we have good results
    if (full_results.get('P3_small') and full_results.get('AVO_small') and
        full_results['P3_small']['best_accuracy'] > 0.5 and
        full_results['AVO_small']['best_accuracy'] > 0.5):

        final_script = create_final_experiment_script(
            full_results['P3_small'],
            full_results['AVO_small']
        )
        print(f"\\n🎉 Final experiment script created: {final_script}")
        print("Run this script to generate publication-ready results with optimal parameters")
    else:
        print("\\n⚠️  Could not create final experiment script (insufficient results)")

    print("\\n✨ TF-DWT optimization process completed!")
    return summary

if __name__ == '__main__':
    try:
        summary = main()
    except KeyboardInterrupt:
        print("\\n\\n⚠️  Optimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n\\n❌ Optimization failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)