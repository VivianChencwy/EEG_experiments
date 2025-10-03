#!/usr/bin/env python3
"""
Generate final comprehensive report of all optimization attempts
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def read_tfdwt_results():
    """Read all TF-DWT result CSVs and organize by timestamp"""
    results = []

    csv_files = sorted(Path('.').glob('tfdwt_summary_stats_*.csv'), key=lambda x: x.stat().st_mtime)

    for csv_file in csv_files:
        timestamp = csv_file.stem.replace('tfdwt_summary_stats_', '')

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            row = list(reader)[0]

            results.append({
                'timestamp': timestamp,
                'file': str(csv_file),
                'overall_acc': float(row['overall_accuracy_mean']),
                'p3_acc': float(row['p3_accuracy_mean']),
                'avo_acc': float(row['avo_accuracy_mean']),
                'p3_std': float(row['p3_accuracy_std']),
                'avo_std': float(row['avo_accuracy_std']),
            })

    return results


def analyze_results(results):
    """Analyze results and identify successful configurations"""
    # Group results by date
    by_date = defaultdict(list)

    for r in results:
        date = r['timestamp'][:8]  # YYYYMMDD
        by_date[date].append(r)

    report = []
    report.append("="*80)
    report.append("TF-DWT OPTIMIZATION FINAL REPORT")
    report.append("="*80)
    report.append("")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total experiments analyzed: {len(results)}")
    report.append("")

    # Overall statistics
    report.append("="*80)
    report.append("OVERALL STATISTICS")
    report.append("="*80)
    report.append("")

    if results:
        all_p3 = [r['p3_acc'] for r in results]
        all_avo = [r['avo_acc'] for r in results]
        all_overall = [r['overall_acc'] for r in results]

        report.append(f"P3 Accuracy:  Mean={sum(all_p3)/len(all_p3):.4f}, Min={min(all_p3):.4f}, Max={max(all_p3):.4f}")
        report.append(f"AVO Accuracy: Mean={sum(all_avo)/len(all_avo):.4f}, Min={min(all_avo):.4f}, Max={max(all_avo):.4f}")
        report.append(f"Overall Acc:  Mean={sum(all_overall)/len(all_overall):.4f}, Min={min(all_overall):.4f}, Max={max(all_overall):.4f}")
        report.append("")

    # Target achievement analysis
    report.append("="*80)
    report.append("TARGET ACHIEVEMENT ANALYSIS")
    report.append("="*80)
    report.append("")

    # Check AVO target (>= 0.66)
    avo_passing = [r for r in results if r['avo_acc'] >= 0.66]
    report.append(f"AVO Target (>= 0.66):")
    report.append(f"  Passing experiments: {len(avo_passing)}/{len(results)}")

    if avo_passing:
        report.append(f"  Best AVO accuracy: {max(r['avo_acc'] for r in avo_passing):.4f}")
        best_avo = max(avo_passing, key=lambda x: x['avo_acc'])
        report.append(f"  Best run: {best_avo['timestamp']}")

    # Check for 5 consecutive AVO passes
    consecutive_avo = 0
    max_consecutive_avo = 0
    for r in results:
        if r['avo_acc'] >= 0.66:
            consecutive_avo += 1
            max_consecutive_avo = max(max_consecutive_avo, consecutive_avo)
        else:
            consecutive_avo = 0

    report.append(f"  Max consecutive passes: {max_consecutive_avo}")
    report.append("")

    # Check P3 target (>= 0.62)
    p3_passing = [r for r in results if r['p3_acc'] >= 0.62]
    report.append(f"P3 Target (>= 0.62):")
    report.append(f"  Passing experiments: {len(p3_passing)}/{len(results)}")

    if p3_passing:
        report.append(f"  Best P3 accuracy: {max(r['p3_acc'] for r in p3_passing):.4f}")
        best_p3 = max(p3_passing, key=lambda x: x['p3_acc'])
        report.append(f"  Best run: {best_p3['timestamp']}")

    # Check for 5 consecutive P3 passes
    consecutive_p3 = 0
    max_consecutive_p3 = 0
    for r in results:
        if r['p3_acc'] >= 0.62:
            consecutive_p3 += 1
            max_consecutive_p3 = max(max_consecutive_p3, consecutive_p3)
        else:
            consecutive_p3 = 0

    report.append(f"  Max consecutive passes: {max_consecutive_p3}")
    report.append("")

    # Daily progress
    report.append("="*80)
    report.append("DAILY PROGRESS")
    report.append("="*80)
    report.append("")

    for date in sorted(by_date.keys()):
        day_results = by_date[date]
        report.append(f"Date: {date}")
        report.append(f"  Experiments: {len(day_results)}")

        day_p3 = [r['p3_acc'] for r in day_results]
        day_avo = [r['avo_acc'] for r in day_results]

        report.append(f"  P3:  Mean={sum(day_p3)/len(day_p3):.4f}, Range=[{min(day_p3):.4f}, {max(day_p3):.4f}]")
        report.append(f"  AVO: Mean={sum(day_avo)/len(day_avo):.4f}, Range=[{min(day_avo):.4f}, {max(day_avo):.4f}]")
        report.append("")

    # Recent results (last 10)
    report.append("="*80)
    report.append("RECENT RESULTS (Last 10)")
    report.append("="*80)
    report.append("")

    for r in results[-10:]:
        report.append(f"{r['timestamp']}: P3={r['p3_acc']:.4f}, AVO={r['avo_acc']:.4f}, Overall={r['overall_acc']:.4f}")

    report.append("")
    report.append("="*80)
    report.append("END OF REPORT")
    report.append("="*80)

    return "\n".join(report)


def main():
    results = read_tfdwt_results()

    if not results:
        print("No TF-DWT results found!")
        return

    report = analyze_results(results)

    # Print to console
    print(report)

    # Save to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f'FINAL_OPTIMIZATION_REPORT_{timestamp}.txt'

    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {report_file}")

    # Save detailed JSON
    json_file = f'detailed_results_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Detailed results saved to: {json_file}")


if __name__ == '__main__':
    main()
