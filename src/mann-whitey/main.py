import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import os

# --- 1. CONFIGURATION ---
# Update these file paths if your CSV files have different names
cwd = os.getcwd()
REFERENCE_CSV = os.path.join(cwd, 'partial_results', 'AF_squim.csv')   # Your 20,000 professional samples
CLINICAL_CSV = os.path.join(cwd, 'partial_results', 'impact_squim.csv')     # Your 1,000 clinical samples

# List of metric columns to analyze
# METRICS_TO_ANALYZE = ['mos_pred', 'noi_pred', 'dis_pred', 'col_pred', 'loud_pred']
# METRICS_TO_ANALYZE = ['mos_pred']
METRICS_TO_ANALYZE = ['STOI', 'PESQ', 'SI-SDR']


# Create a directory to save the plots
PLOTS_OUTPUT_DIR = os.path.join(cwd, 'plots')
if not os.path.exists(PLOTS_OUTPUT_DIR):
    os.makedirs(PLOTS_OUTPUT_DIR)

# --- 2. DATA LOADING AND PREPARATION ---

def load_and_prepare_data(ref_file, clin_file):
    """Loads and combines the two CSVs into a single DataFrame for easy plotting."""
    try:
        ref_df = pd.read_csv(ref_file)
        clin_df = pd.read_csv(clin_file)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure the CSV files are in the same directory as the script.")
        return None

    # Add a 'source' column to identify which dataset each row came from
    ref_df['source'] = 'REDLAT'
    clin_df['source'] = 'Impact'
    
    # Concatenate the two DataFrames into one
    combined_df = pd.concat([ref_df, clin_df], ignore_index=True)
    
    print("Data loaded successfully.")
    print(f"REDLAT samples: {len(ref_df)}")
    print(f"Impact samples:  {len(clin_df)}")
    
    return combined_df

# --- 3. STATISTICAL ANALYSIS ---

def perform_statistical_analysis(df, metrics):
    """Performs a Mann-Whitney U test for each metric and returns a results table."""
    print("\n--- Performing Statistical Analysis (Mann-Whitney U Test) ---")
    
    results = []
    
    # Get the data for each group
    group_ref = df[df['source'] == 'REDLAT']
    group_clin = df[df['source'] == 'Impact']

    for metric in metrics:
        # Extract the specific metric series for each group
        ref_series = group_ref[metric]
        clin_series = group_clin[metric]
        
        # Perform the test
        # We use this test because it doesn't assume a normal distribution, which is safer.
        stat, p_value = mannwhitneyu(ref_series, clin_series, alternative='two-sided')
        
        results.append({
            'Metric': metric,
            'REDLAT Mean': ref_series.mean(),
            'Impact Mean': clin_series.mean(),
            'REDLAT Median': ref_series.median(),
            'Impact Median': clin_series.median(),
            'P-Value': p_value
        })
        
    results_df = pd.DataFrame(results)
    
    # Format the p-value for better readability
    results_df['P-Value (Formatted)'] = results_df['P-Value'].apply(
        lambda p: f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
    )
    results_df['Significant (p<0.05)'] = results_df['P-Value'].apply(lambda p: 'Yes' if p < 0.05 else 'No')
    
    return results_df[['Metric', 'REDLAT Mean', 'Impact Mean', 'P-Value (Formatted)', 'Significant (p<0.05)']]


# --- 4. VISUALIZATION ---

def generate_plots(df, metrics, output_dir):
    """Generates and saves a comparison plot for each metric."""
    print(f"\n--- Generating and saving plots to '{output_dir}' directory ---")
    
    for metric in metrics:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # --- FIXED PLOT 1: Box Plot ---
        # Added hue='source' and legend=False to comply with modern Seaborn standards.
        sns.boxplot(
            x='source', 
            y=metric, 
            hue='source',  # Explicitly assign the variable for coloring
            data=df, 
            ax=axes[0], 
            palette=['skyblue', 'salmon'],
            legend=False   # Hide the redundant legend
        )
        axes[0].set_title(f'Box Plot of {metric}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Dataset', fontsize=12)
        axes[0].set_ylabel('Predicted Score', fontsize=12)
        
        # Plot 2: Density Plot (This one was already correct as it used `hue`)
        sns.kdeplot(data=df, x=metric, hue='source', fill=True, ax=axes[1], palette=['skyblue', 'salmon'])
        axes[1].set_title(f'Density Distribution of {metric}', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted Score', fontsize=12)
        axes[1].set_ylabel('Density', fontsize=12)
        
        # Final touches
        plt.suptitle(f'Comparison of "{metric.upper()}" between Redlat and Impact Audios', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save the figure
        plot_filename = os.path.join(output_dir, f'comparison_{metric}.png')
        plt.savefig(plot_filename, dpi=150)
        plt.close(fig)
        
        print(f"  -> Saved plot: {plot_filename}")

# --- 5. MAIN EXECUTION ---

if __name__ == '__main__':
    # Step 1: Load data
    combined_data = load_and_prepare_data(REFERENCE_CSV, CLINICAL_CSV)
    
    if combined_data is not None:
        # Step 2: Run statistical analysis and print results
        analysis_results = perform_statistical_analysis(combined_data, METRICS_TO_ANALYZE)
        print("\n--- Statistical Results Summary ---")
        print(analysis_results.to_string())
        
        # Step 3: Generate and save all plots
        generate_plots(combined_data, METRICS_TO_ANALYZE, PLOTS_OUTPUT_DIR)
        
        print("\nAnalysis complete. Check the console for the stats table and the output folder for plots.")