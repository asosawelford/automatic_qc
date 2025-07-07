import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def load_reports_to_dataframe(directory: str, category: str) -> pd.DataFrame:
    """Loads all JSON reports from a directory into a pandas DataFrame."""
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith('_report.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                report = json.load(f)

            # We only care about successful reports with analysis results
            if report['processing_status'] == 'SUCCESS' and 'analysis_results' in report:
                res = report['analysis_results']
                flat_data = {
                    'file': report['source_file'],
                    'category': category,
                    'lufs': res['level']['integrated_lufs'],
                    'clipping_%': res['level']['clipping_percent'],
                    'snr_db': res['noise']['snr_db'],
                    'speakers': res['speaker']['estimated_speakers']
                }
                all_data.append(flat_data)

    return pd.DataFrame(all_data)

# --- Main Analysis ---
if __name__ == "__main__":
    # Load data from all categories
    df_gold = load_reports_to_dataframe('/home/aleph/automatic_cq/reports/gold_standard', 'gold')
    df_borderline = load_reports_to_dataframe('/home/aleph/automatic_cq/reports/borderline', 'borderline')
    df_rejected = load_reports_to_dataframe('/home/aleph/automatic_cq/reports/rejected', 'rejected')

    # Combine into a single DataFrame
    df = pd.concat([df_gold, df_borderline, df_rejected], ignore_index=True)

    if df.empty:
        print("No valid reports found. Please run main.py first.")
    else:
        print("--- Combined Data Analysis ---")
        print(df)

        print("\n--- Statistical Summary (Gold Standard Recordings) ---")
        # This is the most important table!
        print(df_gold[['lufs', 'clipping_%', 'snr_db']].describe())

        # --- Visualization ---
        # Create box plots to see the distribution of each metric by category
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        df.boxplot(column='lufs', by='category', ax=axes[0])
        df.boxplot(column='snr_db', by='category', ax=axes[1])
        plt.suptitle('Metric Distributions by Quality Category')
        plt.show()