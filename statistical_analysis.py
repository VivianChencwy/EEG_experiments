"""
Statistical Analysis Tool for EEG Experiment Results

This script performs paired t-tests and other statistical comparisons
between different experimental methods using the detailed CSV results.

Usage:
    python statistical_analysis.py method1_results.csv method2_results.csv

Features:
- Paired t-test for comparing two methods on same folds
- Wilcoxon signed-rank test (non-parametric alternative)
- Effect size calculation (Cohen's d)
- Statistical summary and interpretation
- Support for multiple metrics (accuracy, precision, recall, f1, AUC)
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import sys
import math
from pathlib import Path
from typing import Tuple, Dict, List


def load_and_validate_results(file1: str, file2: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate two result CSV files for comparison."""
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
    except Exception as e:
        raise ValueError(f"Error loading CSV files: {e}")

    # Check if files have the same structure
    required_cols = ['repeat', 'fold', 'overall_accuracy', 'p3_accuracy', 'avo_accuracy']
    missing1 = [col for col in required_cols if col not in df1.columns]
    missing2 = [col for col in required_cols if col not in df2.columns]

    if missing1:
        raise ValueError(f"Method 1 CSV missing columns: {missing1}")
    if missing2:
        raise ValueError(f"Method 2 CSV missing columns: {missing2}")

    # Sort by repeat and fold for proper pairing
    df1 = df1.sort_values(['repeat', 'fold']).reset_index(drop=True)
    df2 = df2.sort_values(['repeat', 'fold']).reset_index(drop=True)

    # Check if they have same number of folds
    if len(df1) != len(df2):
        raise ValueError(f"Different number of results: Method1={len(df1)}, Method2={len(df2)}")

    # Check if repeat/fold pairs match
    if not df1[['repeat', 'fold']].equals(df2[['repeat', 'fold']]):
        raise ValueError("Repeat/fold pairs don't match between methods")

    return df1, df2


def calculate_paired_statistics(values1: np.ndarray, values2: np.ndarray,
                               metric_name: str, alpha: float = 0.05) -> Dict:
    """Calculate comprehensive paired statistics including t-test, Wilcoxon, and effect size."""

    # Calculate differences
    differences = values1 - values2
    n_pairs = len(differences)

    if n_pairs < 2:
        return {
            'metric': metric_name,
            'n_pairs': n_pairs,
            'error': 'Insufficient data for statistical testing'
        }

    # Basic descriptive statistics
    mean1 = np.mean(values1)
    mean2 = np.mean(values2)
    std1 = np.std(values1, ddof=1)
    std2 = np.std(values2, ddof=1)

    # Difference statistics
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    se_diff = std_diff / math.sqrt(n_pairs)

    # Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(values1, values2)

    # Confidence interval for mean difference
    t_crit = stats.t.ppf(1 - alpha/2, df=n_pairs-1)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    # Effect size (Cohen's d for paired samples)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0

    # Wilcoxon signed-rank test (non-parametric)
    try:
        wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(differences, alternative='two-sided')
    except ValueError:
        wilcoxon_stat, wilcoxon_pvalue = None, None

    # Practical interpretation
    def interpret_effect_size(d):
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def interpret_pvalue(p, alpha=0.05):
        if p < alpha:
            return "statistically significant"
        else:
            return "not statistically significant"

    results = {
        'metric': metric_name,
        'n_pairs': n_pairs,
        'method1_mean': mean1,
        'method1_std': std1,
        'method2_mean': mean2,
        'method2_std': std2,
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'se_difference': se_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        't_statistic': t_stat,
        't_pvalue': t_pvalue,
        'cohens_d': cohens_d,
        'effect_size_interpretation': interpret_effect_size(cohens_d),
        'wilcoxon_statistic': wilcoxon_stat,
        'wilcoxon_pvalue': wilcoxon_pvalue,
        't_test_interpretation': interpret_pvalue(t_pvalue, alpha),
        'wilcoxon_interpretation': interpret_pvalue(wilcoxon_pvalue, alpha) if wilcoxon_pvalue else "unavailable"
    }

    return results


def print_statistical_summary(results: Dict, method1_name: str, method2_name: str):
    """Print comprehensive statistical summary."""

    print(f"\n" + "="*80)
    print(f"STATISTICAL COMPARISON: {method1_name} vs {method2_name}")
    print(f"Metric: {results['metric']}")
    print(f"="*80)

    print(f"\nDESCRIPTIVE STATISTICS:")
    print(f"  Number of paired samples: {results['n_pairs']}")
    print(f"  {method1_name:20} Mean ± SD: {results['method1_mean']:.4f} ± {results['method1_std']:.4f}")
    print(f"  {method2_name:20} Mean ± SD: {results['method2_mean']:.4f} ± {results['method2_std']:.4f}")

    print(f"\nPAIRED DIFFERENCE ANALYSIS:")
    print(f"  Mean difference (M1-M2): {results['mean_difference']:.4f}")
    print(f"  Standard error:          {results['se_difference']:.4f}")
    print(f"  95% Confidence interval: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")

    print(f"\nSTATISTICAL TESTS:")
    print(f"  Paired t-test:")
    print(f"    t-statistic: {results['t_statistic']:.4f}")
    print(f"    p-value:     {results['t_pvalue']:.6f}")
    print(f"    Result:      {results['t_test_interpretation']}")

    if results.get('wilcoxon_pvalue') is not None:
        print(f"  Wilcoxon signed-rank test (non-parametric):")
        print(f"    W-statistic: {results['wilcoxon_statistic']:.1f}")
        print(f"    p-value:     {results['wilcoxon_pvalue']:.6f}")
        print(f"    Result:      {results['wilcoxon_interpretation']}")

    print(f"\nEFFECT SIZE:")
    print(f"  Cohen's d:    {results['cohens_d']:.4f}")
    print(f"  Magnitude:    {results['effect_size_interpretation']}")

    print(f"\nINTERPRETATION:")
    if results['t_pvalue'] < 0.05:
        direction = "better" if results['mean_difference'] > 0 else "worse"
        print(f"  {method1_name} performs significantly {direction} than {method2_name}")
        print(f"  with a {results['effect_size_interpretation']} effect size.")
    else:
        print(f"  No statistically significant difference between methods.")
        print(f"  Effect size is {results['effect_size_interpretation']}.")


def analyze_multiple_metrics(df1: pd.DataFrame, df2: pd.DataFrame,
                           method1_name: str, method2_name: str) -> pd.DataFrame:
    """Analyze multiple metrics and return summary DataFrame."""

    metrics_to_analyze = [
        'overall_accuracy', 'p3_accuracy', 'avo_accuracy',
        'p3_precision', 'p3_recall', 'p3_f1', 'p3_auc',
        'avo_precision', 'avo_recall', 'avo_f1', 'avo_auc'
    ]

    summary_results = []

    for metric in metrics_to_analyze:
        if metric in df1.columns and metric in df2.columns:
            values1 = df1[metric].values
            values2 = df2[metric].values

            # Remove any NaN values
            valid_mask = ~(np.isnan(values1) | np.isnan(values2))
            values1_clean = values1[valid_mask]
            values2_clean = values2[valid_mask]

            if len(values1_clean) > 1:
                results = calculate_paired_statistics(values1_clean, values2_clean, metric)

                # Print detailed results
                print_statistical_summary(results, method1_name, method2_name)

                # Add to summary
                summary_results.append({
                    'metric': metric,
                    'n_pairs': results['n_pairs'],
                    'method1_mean': results['method1_mean'],
                    'method2_mean': results['method2_mean'],
                    'mean_difference': results['mean_difference'],
                    't_statistic': results['t_statistic'],
                    't_pvalue': results['t_pvalue'],
                    'cohens_d': results['cohens_d'],
                    'effect_size': results['effect_size_interpretation'],
                    'significant': results['t_pvalue'] < 0.05
                })

    return pd.DataFrame(summary_results)


def main():
    if len(sys.argv) != 3:
        print("Usage: python statistical_analysis.py method1_results.csv method2_results.csv")
        print("\nExample:")
        print("  python statistical_analysis.py tfdwt_detailed_results_20241123_143022.csv baseline_detailed_results_20241123_141505.csv")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    # Extract method names from filenames
    method1_name = Path(file1).stem.split('_')[0].upper()
    method2_name = Path(file2).stem.split('_')[0].upper()

    try:
        print(f"Loading results from:")
        print(f"  Method 1 ({method1_name}): {file1}")
        print(f"  Method 2 ({method2_name}): {file2}")

        # Load and validate data
        df1, df2 = load_and_validate_results(file1, file2)

        print(f"\nSuccessfully loaded {len(df1)} paired results")
        print(f"Repeats: {df1['repeat'].nunique()}, Folds per repeat: {df1['fold'].nunique()}")

        # Perform comprehensive analysis
        summary_df = analyze_multiple_metrics(df1, df2, method1_name, method2_name)

        # Save summary to CSV
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        summary_filename = f'statistical_comparison_{method1_name}_vs_{method2_name}_{timestamp}.csv'
        summary_df.to_csv(summary_filename, index=False)

        print(f"\n" + "="*80)
        print(f"SUMMARY OF ALL METRICS")
        print(f"="*80)
        print(summary_df.to_string(index=False, float_format='%.4f'))

        print(f"\nDetailed summary saved to: {summary_filename}")

        # Highlight significant results
        significant_results = summary_df[summary_df['significant']]
        if len(significant_results) > 0:
            print(f"\nSIGNIFICANT DIFFERENCES FOUND:")
            for _, row in significant_results.iterrows():
                direction = "higher" if row['mean_difference'] > 0 else "lower"
                print(f"  {row['metric']}: {method1_name} {direction} by {abs(row['mean_difference']):.4f} (p={row['t_pvalue']:.4f}, d={row['cohens_d']:.3f})")
        else:
            print(f"\nNo statistically significant differences found between {method1_name} and {method2_name}")

    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()