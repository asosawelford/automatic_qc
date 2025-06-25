import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import percentileofscore
from pathlib import Path

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

    Args:
        reference_df (pd.DataFrame): The reference (e.g., clinical) dataset.
        target_df (pd.DataFrame): The target dataset to compare against the reference.
        metrics_list (list): A list of column names for the metrics to compare.
        output_dir (Path): The directory where all results (CSVs, plots) will be saved.
        reference_name (str): A label for the reference dataset used in plots and reports.
        target_name (str): A label for the target dataset used in plots and reports.
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
        
        # --- THIS IS THE CORRECTED LINE ---
        dynamic_palette = {reference_name: 'skyblue', target_name: 'lightcoral'}
        sns.violinplot(y='value', x='dataset', data=plot_df, palette=dynamic_palette, inner='quartile', ax=ax2)
        # ------------------------------------
        
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
    summary_df = pd.DataFrame(summary_stats_list).reset_index().rename(columns={'index': 'statistic'})
    stat_order = ['metric', 'dataset', 'statistic', 'count', 'mean', 'std', 'min', '5%', '25%', '50%', '75%', '95%', 'max']
    cols_to_select = [col for col in stat_order if col in summary_df.columns]
    summary_df = summary_df[cols_to_select]
    summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)

    individual_results_df.to_csv(output_dir / 'target_vs_reference_individual_metrics.csv', index=False)

    print(f"Analysis complete. All results saved to an output directory: '{output_dir}'")


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

    # --- 3. Run Comparison ---
    compare_audio_datasets(
        reference_df=reference_dataset,
        target_df=target_dataset,
        metrics_list=METRICS_TO_COMPARE,
        output_dir=OUTPUT_DIRECTORY,
        reference_name=REFERENCE_NAME,
        target_name=TARGET_NAME
    )