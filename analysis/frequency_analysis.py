import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load Data ---

# Load difficulty labels (has 'index', 'category', etc.)
difficulty_df = pd.read_csv("analysis/performance_results/sample_performance_summary.csv")
print(difficulty_df.head())

# Load spectrogram features (has 'index' or 'file', plus feature columns)
features_df = pd.read_csv("analysis/performance_results/dev_spectrogram_features.csv")
print(features_df.head())

# Merge on sample index
merged_df = features_df.merge(difficulty_df, on="index", how="inner")

# --- Quick overview ---
print(merged_df["category"].value_counts())
print(merged_df.groupby("category")[[
    "energy_mean", "main_mel_mean", "spectral_centroid_mean"
]].describe())


# Statistical comparison

from scipy.stats import kruskal

easy = merged_df[merged_df["category"] == "easy"]["energy_mean"]
hard = merged_df[merged_df["category"] == "hard"]["energy_mean"]
unstable = merged_df[merged_df["category"] == "unstable"]["energy_mean"]

stat, p = kruskal(easy, hard, unstable)
print(f"Kruskal-Wallis H-test: stat={stat:.3f}, p={p:.4f}")


# --- Visualize distributions ---

# Plot distribution of energy per category
plt.figure(figsize=(10, 5))
sns.boxplot(data=merged_df, x="category", y="energy_mean")
plt.title("Energy Mean per Difficulty Category")
plt.tight_layout()
plt.savefig("analysis/energy_mean_per_cat.png")
plt.close()

# Plot main mel bin
plt.figure(figsize=(10, 5))
# sns.violinplot(data=merged_df, x="category", y="main_mel_mean", inner="quartile")
# sns.violinplot(data=merged_df, x="category", y="main_mel_mean", inner=None, color=".8")
sns.boxplot(data=merged_df, x="category", y="main_mel_mean")
plt.title("Main Mel Bin Mean per Difficulty Category")
plt.tight_layout()
plt.savefig("analysis/main_mel_mean_per_cat.png")
plt.close()

# Spectral centroid
plt.figure(figsize=(10, 5))
sns.boxplot(data=merged_df, x="category", y="spectral_centroid_mean")
plt.title("Spectral Centroid Mean per Difficulty Category")
plt.tight_layout()
plt.savefig("analysis/centroid_mean_per_cat.png")
plt.close()
