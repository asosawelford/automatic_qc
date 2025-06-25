import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import percentileofscore
from pathlib import Path

# (The compare_audio_datasets function from the previous step goes here. No changes needed.)
def compare_audio_datasets(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    metrics_list: list,
    output_dir: Path,
    reference_name: str = 'Reference',
    target_name: str = 'Target'
):
    """
    Compares two datasets on specified audio quality metrics, saves summary
    statistics, individual sample comparisons, and plots to an output directory.
    """
    # --- 1. Setup and Validation ---
    plot_dir = output_dir / 'plots'
    output_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    for df, name in [(reference_df, reference_name), (target_df, target_name)]:
        missing_cols = [m for m in metrics_list if m not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset '{name}' is missing required metric columns: {missing_cols}")

    summary_stats_list = []
    individual_results_df = target_df.copy()

    # --- 2. Process Each Metric ---
    print(f"Processing {len(metrics_list)} metrics...")
    for metric in metrics_list:
        ref_stats = reference_df[metric].describe(percentiles=[.05, .25, .50, .75, .95])
        ref_stats['dataset'] = reference_name
        ref_stats['metric'] = metric
        summary_stats_list.append(ref_stats)

        target_stats = target_df[metric].describe(percentiles=[.05, .25, .50, .75, .95])
        target_stats['dataset'] = target_name
        target_stats['metric'] = metric
        summary_stats_list.append(target_stats)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        sns.histplot(reference_df[metric], kde=True, stat='density', color='blue', label=reference_name, alpha=0.5, bins=30, ax=ax1)
        sns.histplot(target_df[metric], kde=True, stat='density', color='red', label=target_name, alpha=0.5, bins=30, ax=ax1)
        ax1.axvline(ref_stats['mean'], color='blue', linestyle='--', linewidth=1.5, label=f'{reference_name} Mean')
        ax1.axvline(target_stats['mean'], color='red', linestyle='--', linewidth=1.5, label=f'{target_name} Mean')
        ax1.set_title(f'Distribution Comparison for {metric}')
        ax1.set_xlabel(metric)
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        plot_df = pd.DataFrame({
            'value': pd.concat([reference_df[metric], target_df[metric]], ignore_index=True),
            'dataset': [reference_name] * len(reference_df) + [target_name] * len(target_df)
        })
        
        dynamic_palette = {reference_name: 'skyblue', target_name: 'lightcoral'}
        sns.violinplot(y='value', x='dataset', data=plot_df, palette=dynamic_palette, inner='quartile', ax=ax2)
        
        ax2.set_title(f'Violin Plot Comparison for {metric}')
        ax2.set_ylabel(metric)
        ax2.set_xlabel('Dataset')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(plot_dir / f'{metric}_comparison.png')
        plt.close(fig)

        ref_mean = ref_stats['mean']
        ref_std = ref_stats['std']

        if ref_std > 1e-9:
            z_scores = (target_df[metric] - ref_mean) / ref_std
        else:
            z_scores = np.nan
        individual_results_df[f'{metric}_z_score_vs_ref'] = z_scores

        percentile_ranks = target_df[metric].apply(
            lambda x: percentileofscore(reference_df[metric], x, kind='weak')
        )
        individual_results_df[f'{metric}_percentile_rank_vs_ref'] = percentile_ranks

    # --- 3. Save Final Results to CSV ---
    summary_df = pd.DataFrame(summary_stats_list)
    summary_df.reset_index(drop=True, inplace=True) # Drop the old index, we already have 'metric' and 'dataset' columns

    # Reorder columns for a clean, final CSV file
    cols_order = ['metric', 'dataset', 'count', 'mean', 'std', 'min', '5%', '25%', '50%', '75%', '95%', 'max']
    # Filter for columns that actually exist to prevent errors
    existing_cols = [col for col in cols_order if col in summary_df.columns]
    summary_df = summary_df[existing_cols]

    summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)


    individual_results_df.to_csv(output_dir / 'target_vs_reference_individual_metrics.csv', index=False)

    print(f"Analysis complete. All results saved to an output directory: '{output_dir}'")

# (New function goes here)
def generate_summary_report(
    summary_stats_path: Path,
    individual_results_path: Path,
    output_dir: Path,
    metrics_list: list,
    reference_name: str,
    target_name: str
):
    """
    Reads the generated CSVs and creates a human-readable text report
    summarizing the comparison.
    """
    try:
        summary_df = pd.read_csv(summary_stats_path)
        individual_df = pd.read_csv(individual_results_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find required input files for report generation. {e}")
        return

    report_lines = []
    report_lines.append(f"Audio Quality Comparison Report: '{target_name}' vs. '{reference_name}'")
    report_lines.append("=" * 80)
    report_lines.append("This report assumes that for all metrics, a higher score indicates better quality.")
    report_lines.append(f"Total samples in '{target_name}': {len(individual_df)}")
    ref_count = int(summary_df.loc[summary_df['dataset'] == reference_name, 'count'].iloc[0])
    report_lines.append(f"Total samples in '{reference_name}': {ref_count}")

    for metric in metrics_list:
        report_lines.append("\n" + "-" * 80)
        report_lines.append(f"METRIC SUMMARY: {metric.upper()}")
        report_lines.append("-" * 80)

        # --- Aggregate Comparison ---
        ref_metric_stats = summary_df[(summary_df['dataset'] == reference_name) & (summary_df['metric'] == metric)].iloc[0]
        target_metric_stats = summary_df[(summary_df['dataset'] == target_name) & (summary_df['metric'] == metric)].iloc[0]
        
        report_lines.append("[Aggregate Performance]")
        report_lines.append(f"  - Mean Score:   {target_name} ({target_metric_stats['mean']:.2f}) vs. {reference_name} ({ref_metric_stats['mean']:.2f})")
        report_lines.append(f"  - Median Score: {target_name} ({target_metric_stats['50%']:.2f}) vs. {reference_name} ({ref_metric_stats['50%']:.2f})")

        # --- Individual Sample Distribution vs. Reference ---
        percentile_col = f'{metric}_percentile_rank_vs_ref'
        if percentile_col not in individual_df.columns:
            continue
            
        percentile_ranks = individual_df[percentile_col]
        total_samples = len(percentile_ranks)

        n_in_iqr = ((percentile_ranks >= 25) & (percentile_ranks <= 75)).sum()
        pct_in_iqr = (n_in_iqr / total_samples) * 100

        n_above_median = (percentile_ranks > 50).sum()
        pct_above_median = (n_above_median / total_samples) * 100

        n_above_q3 = (percentile_ranks > 75).sum()
        pct_above_q3 = (n_above_q3 / total_samples) * 100

        report_lines.append(f"\n[Distribution of '{target_name}' Samples vs. '{reference_name}' Population]")
        report_lines.append(f"  - Samples within Reference IQR (25th-75th percentile):   {n_in_iqr:d} ({pct_in_iqr:.1f}%)")
        report_lines.append(f"  - Samples better than Reference Median (>50th percentile):  {n_above_median:d} ({pct_above_median:.1f}%)")
        report_lines.append(f"  - Samples in Reference Top Quartile (>75th percentile):   {n_above_q3:d} ({pct_above_q3:.1f}%)")
        
        # --- Automated Conclusion ---
        conclusion = "Overall, for this metric, the quality of the '{target}' dataset appears ".format(target=target_name)
        if target_metric_stats['mean'] > ref_metric_stats['mean'] and pct_above_q3 > 35:
            conclusion += f"**significantly better** than the '{reference_name}' dataset."
        elif target_metric_stats['mean'] > ref_metric_stats['mean'] and pct_above_median > 50:
            conclusion += f"**generally better** than the '{reference_name}' dataset."
        elif abs(target_metric_stats['mean'] - ref_metric_stats['mean']) < (ref_metric_stats['std'] * 0.25) and pct_in_iqr > 40:
             conclusion += f"**highly comparable** to the '{reference_name}' dataset."
        elif target_metric_stats['mean'] < ref_metric_stats['mean']:
             conclusion += f"**lower** than the '{reference_name}' dataset."
        else:
            conclusion += f"to have a complex relationship with the '{reference_name}' dataset that requires further review."

        report_lines.append("\n[Conclusion]")
        report_lines.append(f"  {conclusion}")

    report_path = output_dir / 'final_summary_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Generated summary report: {report_path}")

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # --- 1. Configuration ---
    BASE_DIR = Path.cwd()
    REFERENCE_DATA_PATH = BASE_DIR / 'metrics' / 'redlat.csv'
    TARGET_DATA_PATH = BASE_DIR / 'metrics' / 'impact.csv'


    REFERENCE_NAME = 'REDLAT'
    TARGET_NAME = 'IMPACT'

    METRICS_TO_COMPARE = ['MOS', 'PESQ', 'STOI', 'SI-SDR']


    OUTPUT_DIRECTORY = BASE_DIR / 'quality_comparison_report'
    
    # --- 2. Load Data ---
    print(f"Loading reference dataset: {REFERENCE_DATA_PATH}")
    reference_dataset = pd.read_csv(REFERENCE_DATA_PATH)
    
    print(f"Loading target dataset: {TARGET_DATA_PATH}")
    target_dataset = pd.read_csv(TARGET_DATA_PATH)

    # --- 3. Run Comparison and save CSVs/plots ---
    compare_audio_datasets(
        reference_df=reference_dataset,
        target_df=target_dataset,
        metrics_list=METRICS_TO_COMPARE,
        output_dir=OUTPUT_DIRECTORY,
        reference_name=REFERENCE_NAME,
        target_name=TARGET_NAME
    )
    
    # --- 4. Generate the final text report from the saved CSVs ---
    generate_summary_report(
        summary_stats_path=OUTPUT_DIRECTORY / 'summary_statistics.csv',
        individual_results_path=OUTPUT_DIRECTORY / 'target_vs_reference_individual_metrics.csv',
        output_dir=OUTPUT_DIRECTORY,
        metrics_list=METRICS_TO_COMPARE,
        reference_name=REFERENCE_NAME,
        target_name=TARGET_NAME
    )
