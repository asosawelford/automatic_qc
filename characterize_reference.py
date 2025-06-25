"""
for each quality metric:
Calculate key descriptive statistics (mean, median, standard deviation, min, max, and specific percentiles).
Generate a histogram with a Kernel Density Estimate (KDE) to show the distribution shape.
Generate a box plot to visualize central tendency, spread, and potential outliers.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cwd = os.getcwd()
clinical_dataset = pd.read_csv(os.path.join(cwd, 'partial_results', 'redlat_results.csv'))
metrics = ['mos_pred', 'dis_pred', 'col_pred', 'loud_pred', 'noi_pred']

# --- 2. FUNCTION TO CHARACTERIZE THE DATASET ---

def characterize_clinical_dataset(df: pd.DataFrame, metrics_list: list):
    """
    Analyzes and visualizes the distribution of specified audio quality metrics
    within a clinical dataset.

    Args:
        df (pd.DataFrame): The DataFrame containing your clinical dataset with
                           audio quality metrics as columns.
        metrics_list (list): A list of strings, where each string is the name
                             of a metric column in the DataFrame to analyze.
    """
    print("--- Characterizing Clinical Dataset ---")
    print(f"Total recordings in dataset: {len(df)}\n")

    # Store statistics for easy reference
    all_stats = {}

    for metric in metrics_list:
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in DataFrame columns. Skipping.")
            continue

        print(f"\n--- Metric: {metric} ---")

        # --- Descriptive Statistics ---
        # Using .describe() for quick overview, then extracting specific percentiles
        desc_stats = df[metric].describe(percentiles=[.05, .25, .50, .75, .95])
        all_stats[metric] = desc_stats.to_dict()

        print("Descriptive Statistics:")
        print(desc_stats.to_string()) # .to_string() for better console formatting

        # Calculate IQR
        Q1 = df[metric].quantile(0.25)
        Q3 = df[metric].quantile(0.75)
        IQR = Q3 - Q1
        print(f"  IQR (Interquartile Range): {IQR:.4f}")

        # --- Visualizations ---
        plt.figure(figsize=(12, 5))

        # Histogram with KDE
        plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
        sns.histplot(df[metric], kde=True, bins=30, color='skyblue')
        plt.title(f'Distribution of {metric}\n(Histogram with KDE)')
        plt.xlabel(metric)
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Box Plot
        plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
        sns.boxplot(y=df[metric], color='lightcoral')
        plt.title(f'Box Plot of {metric}')
        plt.ylabel(metric)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout() # Adjust subplot parameters for a tight layout
        plt.savefig(os.path.join(cwd,'plots', f'{metric}'))

        print("\nInterpretation for this metric:")
        print(f"  - Mean: {all_stats[metric]['mean']:.3f}, Median: {all_stats[metric]['50%']:.3f}")
        print(f"  - Std Dev: {all_stats[metric]['std']:.3f} (Lower indicates less variability)")
        print(f"  - Range: {all_stats[metric]['min']:.3f} to {all_stats[metric]['max']:.3f}")
        print(f"  - 5th Percentile: {all_stats[metric]['5%']:.3f} (Only 5% of recordings are below this value)")
        print(f"  - 95th Percentile: {all_stats[metric]['95%']:.3f} (Only 5% of recordings are above this value)")
        print("  - The histogram shows the overall shape of the distribution.")
        print("  - The box plot identifies the median (middle line), IQR (box), and potential outliers (points outside whiskers).")
        
        # Add a note about direction of quality
        if metric in ['dis_pred', 'col_pred', 'noi_pred']:
            print("  - For this metric, typically *lower* values indicate *higher* quality.")
        elif metric == 'mos_pred':
            print("  - For this metric, typically *higher* values indicate *higher* quality.")
        elif metric == 'loud_pred':
            print("  - For this metric, a specific *range* of values (not too loud, not too quiet) typically indicates optimal quality.")


    print("\n--- Clinical Dataset Characterization Complete ---")
    print("\nThese statistics and visualizations are your baseline to compare your sample against.")
    print("Keep these in mind for Step 2: Comparing your sample.")

    return all_stats # Return statistics dictionary for potential later use

# --- 3. RUN THE CHARACTERIZATION ---
dataset_stats = characterize_clinical_dataset(clinical_dataset, metrics)

# You can access the calculated statistics like this:
# print("\nAccessing Loudness stats:", dataset_stats['Loudness'])