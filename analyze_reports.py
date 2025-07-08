import os
import sys
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # For better plotting of discrete data
from matplotlib.ticker import PercentFormatter

def load_reports_to_dataframe(directory: str, category_name: str) -> pd.DataFrame:
    """Loads all JSON reports from a directory into a pandas DataFrame."""
    if not os.path.isdir(directory):
        print(f"Warning: Directory not found, skipping: {directory}")
        return pd.DataFrame()

    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith('_report.json'):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r') as f:
                    report = json.load(f)
                
                if report['processing_status'] == 'SUCCESS' and 'analysis_results' in report:
                    res = report['analysis_results']
                    flat_data = {
                        'file': report['source_file'],
                        'category': category_name,
                        'lufs': res['level'].get('integrated_lufs'),
                        'clipping_%': res['level'].get('clipping_percent'),
                        'snr_db': res['noise'].get('snr_db'),
                        'speakers': res['speaker'].get('estimated_speakers')
                    }
                    all_data.append(flat_data)
            except Exception as e:
                print(f"Warning: Could not read or parse file {filepath}: {e}")

    return pd.DataFrame(all_data)

def main(report_dirs: list):
    """
    Main function to load, analyze, and plot data from report directories.
    """
    # --- Data Loading (same as before) ---
    all_dfs = []
    for directory in report_dirs:
        category_name = os.path.basename(os.path.normpath(directory))
        df = load_reports_to_dataframe(directory, category_name)
        all_dfs.append(df)
        
    combined_df = pd.concat(all_dfs, ignore_index=True)

    if combined_df.empty:
        print("No valid reports found in the specified directories. Exiting.")
        return

    print("--- Combined Data Analysis ---")
    print(combined_df)
    
    # --- Statistical Summary for each category ---
    print("\n--- Statistical Summaries by Category ---")
    for category in combined_df['category'].unique():
        print(f"\n--- {category.upper()} ---")
        category_df = combined_df[combined_df['category'] == category]
        
        # General stats
        print(category_df[['lufs', 'clipping_%', 'snr_db']].describe())
        
        # --- ADDITION: Print raw speaker counts for context ---
        print("\nSpeaker Counts (Absolute):")
        print(category_df['speakers'].value_counts().sort_index())

    # --- Visualization ---
    plot_config = {
        'lufs': {'type': 'boxplot', 'ylim': (-40, -10), 'title': 'Loudness (LUFS)'},
        'snr_db': {'type': 'boxplot', 'ylim': (-5, 30), 'title': 'Signal-to-Noise Ratio (dB)'},
        'speakers': {'type': 'normalized_barplot', 'title': 'Proportion of Speaker Counts'},
        'clipping_%': {'type': 'boxplot', 'ylim': (-0.001, 0.01), 'title': 'Clipping Percentage'}
    }

    scaler = 2

    TITLE_FONTSIZE, AXIS_LABEL_FONTSIZE, TICK_FONTSIZE, SUPTITLE_FONTSIZE = 0*scaler, 12*scaler, 10*scaler, 20*scaler
    metrics_to_plot, num_metrics = list(plot_config.keys()), len(plot_config.keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    plt.style.use('seaborn-v0_8-whitegrid')
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        config = plot_config[metric]

        if config['type'] == 'normalized_barplot':
            # --- THE FIX: Calculate proportions instead of absolute counts ---
            # Group by category, get speaker value counts, normalize to get proportions
            norm_df = combined_df.groupby('category')['speakers'].value_counts(normalize=True)
            norm_df = norm_df.mul(100).rename('percentage').reset_index()

            sns.barplot(x='speakers', y='percentage', hue='category', data=norm_df, ax=ax, palette="Set2")
            
            ax.set_xlabel('Number of Speakers', fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylabel('Percentage of Files', fontsize=AXIS_LABEL_FONTSIZE)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100)) # Format y-axis as percentage
            ax.get_legend().set_title('Category')

        else: # 'boxplot'
            sns.boxplot(x='category', y=metric, data=combined_df, ax=ax, palette="Set2")
            if config.get('ylim'):
                ax.set_ylim(config['ylim'])
            ax.set_xlabel('Category', fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylabel(metric.upper(), fontsize=AXIS_LABEL_FONTSIZE)

        ax.set_title(config['title'], fontsize=TITLE_FONTSIZE)
        ax.tick_params(axis='x', labelsize=TICK_FONTSIZE, rotation=10)
        ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)

    for j in range(num_metrics, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Metric Distributions by Quality Category', fontsize=SUPTITLE_FONTSIZE, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze and visualize quality assessment reports from multiple directories."
    )
    parser.add_argument(
        'directories', 
        nargs='+', # This allows one or more arguments
        help="One or more directories containing the JSON report files."
    )
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    main(args.directories)