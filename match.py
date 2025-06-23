import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu


cwd = os.getcwd()

# ==============================================================================
# 1. SETUP AND CONFIGURATION
# ==============================================================================

REFERENCE_CSV = os.path.join(cwd, 'partial_results', 'NISQA_results.csv')   # Your 20,000 professional samples
CLINICAL_CSV = os.path.join(cwd, 'partial_results', 'impact_results.csv')     # Your 1,000 clinical samples

METRICS_TO_MATCH = ['noi_pred', 'dis_pred', 'col_pred', 'loud_pred']
PERCENT_TO_KEEP = 100  # <-- The knob you can tune

# ==============================================================================
# 2. FUNCTION TO GET MATCHED DATA (Encapsulated and Safe)
# ==============================================================================
def get_matched_subset(ref_df, clin_df, metrics, percent_to_keep):
    """
    Calculates Mahalanobis distance and returns a matched subset of the clinical data.
    Works on copies to prevent side effects.
    """
    # Defensive Copying: Work on copies to avoid modifying the original DataFrames
    ref_copy = ref_df.copy()
    clin_copy = clin_df.copy()

    print(f"Original clinical dataset size: {len(clin_copy)}")
    
    # --- Calculate Mahalanobis Distance ---
    ref_metrics = ref_copy[metrics].values
    clin_metrics = clin_copy[metrics].values
    mean_vector = np.mean(ref_metrics, axis=0)
    inv_cov_matrix = np.linalg.inv(np.cov(ref_metrics, rowvar=False))
    
    mahalanobis_dist = cdist(clin_metrics, [mean_vector], metric='mahalanobis', VI=inv_cov_matrix)[:, 0]
    clin_copy['mahalanobis_dist'] = mahalanobis_dist

    # --- Pragmatic Percentile Filtering ---
    if percent_to_keep >= 100:
        print("Keeping 100% of the data. No filtering applied.")
        return clin_copy # Return the full dataframe with the distance column
        
    cutoff_distance = np.percentile(mahalanobis_dist, percent_to_keep)
    print(f"\nKeeping the best {percent_to_keep}% of clinical samples.")
    print(f"Calculated Mahalanobis distance cutoff: {cutoff_distance:.4f}")

    matched_subset = clin_copy[clin_copy['mahalanobis_dist'] < cutoff_distance].copy()
    
    outlier_count = len(clin_copy) - len(matched_subset)
    print(f"Removed {outlier_count} clinical samples.")
    print(f"New matched clinical dataset size: {len(matched_subset)}")

    return matched_subset

# ==============================================================================
# 3. FUNCTION TO RUN AND VERIFY ANALYSIS (Encapsulated and Safe)
# ==============================================================================
def run_and_verify_analysis(ref_df, clin_df_subset, metrics):
    """
    Runs the Mann-Whitney U test and prints the results for a given clinical subset.
    """
    print("\n--- Verifying the new matched dataset by re-running analysis ---")
    
    # Defensive Copying again
    ref_copy = ref_df.copy()
    clin_copy = clin_df_subset.copy()

    # The verification test
    for metric in metrics:
        # We can now directly compare the two dataframes without building a combined one
        ref_series = ref_copy[metric]
        clin_series = clin_copy[metric]
        
        # This is a critical debug step. Check if the series are valid.
        if clin_series.empty:
            print(f"Metric: {metric:<10} | Clinical series is empty! Skipping test.")
            continue
            
        _, p_value = mannwhitneyu(ref_series, clin_series, alternative='two-sided')
        is_significant = 'Yes' if p_value < 0.05 else 'No'
        
        print(f"Metric: {metric:<10} | New P-Value: {p_value:.4f} | Still Significant (p<0.05)? {is_significant}")

# ==============================================================================
# 4. MAIN EXECUTION SCRIPT
# ==============================================================================
if __name__ == '__main__':
    try:
        # Load fresh data from disk
        original_ref_df = pd.read_csv(REFERENCE_CSV)
        original_clin_df = pd.read_csv(CLINICAL_CSV)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure CSV files are present.")
        exit()
        
    # --- First, run the analysis on the ORIGINAL data to see the baseline ---
    print("=" * 60)
    print("ANALYSIS ON ORIGINAL (UNFILTERED) DATA")
    print("=" * 60)
    run_and_verify_analysis(original_ref_df, original_clin_df, METRICS_TO_MATCH)
    
    # --- Now, get the matched subset ---
    print("\n" + "=" * 60)
    print(f"GENERATING MATCHED SUBSET (KEEPING {PERCENT_TO_KEEP}%)")
    print("=" * 60)
    matched_clinical_subset = get_matched_subset(
        original_ref_df, 
        original_clin_df, 
        metrics=METRICS_TO_MATCH, 
        percent_to_keep=PERCENT_TO_KEEP
    )

    # --- Finally, run the analysis on the new MATCHED data ---
    if not matched_clinical_subset.empty:
        print("\n" + "=" * 60)
        print("ANALYSIS ON NEW (MATCHED) DATA")
        print("=" * 60)
        run_and_verify_analysis(original_ref_df, matched_clinical_subset, METRICS_TO_MATCH)
