import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300
plt.rcParams["figure.autolayout"] = True


# List of your dataset names and their corresponding CSV file paths
datasets = {
    'ReDLat': '/home/aleph/automatic_cq/impact-grant/metrics/ReDLat.csv',
    'Impact': '/home/aleph/automatic_cq/impact-grant/metrics/Impact.csv',
    'ADReSSo': '/home/aleph/automatic_cq/impact-grant/metrics/ADReSSo.csv'
}

all_data = []

for name, filepath in datasets.items():
    try:
        df = pd.read_csv(filepath)
        # Ensure the 'MOS' column exists
        if 'MOS' in df.columns:
            df_mos = df[['MOS']].copy() # Select only the 'MOS' column
            df_mos['Dataset'] = name   # Add a column to identify the source dataset
            all_data.append(df_mos)
        else:
            print(f"Warning: 'MOS' column not found in {filepath}. Skipping.")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}. Please check the path and filename.")
    except Exception as e:
        print(f"An error occurred while reading {filepath}: {e}")

# Concatenate all individual DataFrames into one large DataFrame
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    print("\nCombined DataFrame head:")
    print(combined_df.head())
    print("\nCombined DataFrame info:")
    combined_df.info()
else:
    print("No data was loaded. Exiting.")
    exit() # Exit if no data was successfully loaded


# --- 3. Calculate Statistics (Mean and Standard Deviation) ---

# Group by the 'Dataset' column and calculate mean and std for 'MOS'
summary_stats = combined_df.groupby('Dataset')['MOS'].agg(['mean', 'std']).reset_index()

print("\nSummary Statistics (Mean and Std Dev of MOS per Dataset):")
print(summary_stats)


# --- 4. Plotting with Seaborn (Violin Plot) ---

plt.figure(figsize=(10, 7))

# Create the violin plots
# 'x' is the categorical variable (Dataset name)
# 'y' is the numerical variable (MOS scores)
# 'inner="quartile"' or "box" shows quartiles (25th, 50th/median, 75th percentiles)
# You can also use 'inner="point"' to show individual data points
ax = sns.violinplot(x='Dataset', y='MOS', data=combined_df, inner="quartile", palette="viridis")
# ax = sns.violinplot(x='Dataset', y='MOS', data=combined_df, inner=None, palette="viridis")

# --- 5. Overlay Mean and Standard Deviation ---

# It's a bit tricky to directly plot mean/std using 'inner' in violinplot
# for an explicit bar. The best way is often to overlay scatter/error points.

# Iterate through the summary statistics to plot mean and SD for each dataset
for i, row in summary_stats.iterrows():
    # Find the x-coordinate for the current dataset.
    # Seaborn automatically maps categories to 0, 1, 2...
    # We can rely on the order of 'Dataset' in summary_stats matching the plot order.
    x_pos = ax.get_xticks()[i] # Get the x-tick position for the i-th category label

    # Plot the mean as a large white dot
    plt.scatter(x_pos, row['mean'], color='red', marker='o', s=100, zorder=5, label='Mean' if i == 0 else "") # label for legend only once

    # Plot the standard deviation as a red error bar around the mean
    # yerr takes a scalar or array. Here, it's the standard deviation.
    plt.errorbar(x_pos, row['mean'], yerr=row['std'], color='red', fmt='none', capsize=5, lw=2, zorder=4, label='Std. Dev.' if i == 0 else "")


# --- Customize the plot ---
plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Overall Speech Quality', fontsize=12)
plt.ylim(0.0, 4.8) # Set y-axis limits to clearly show 1-5 MOS scale, with little buffer
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add a legend for the mean and standard deviation markers
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles)) # Deduplicate labels if any
plt.legend(by_label.values(), by_label.keys(), title="Statistics", loc='upper left')

plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()

# You can also save the figure
# plt.savefig('mos_distribution_violin_plot.png', dpi=300)
# plt.savefig('mos_distribution_violin_plot.pdf')
print("Plot displayed successfully.")