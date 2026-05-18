import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, recall_score, precision_score

# relevant columns (the targets)
supercategories = ["Anth", "Bio", "Geo", "Sil"]

def f1_for_group(group_pred, group_gt):
    return f1_score(group_gt, group_pred, average="macro", zero_division=0)

def f1_for_group_all_classes(group_pred, group_gt):
    scores = f1_score(group_gt, group_pred, average=None, zero_division=0)
    return dict(zip(supercategories, scores))

# def metrics_for_group_all_classes(group_pred, group_gt):
#     f1s = f1_score(group_gt, group_pred, average=None, zero_division=0)
#     precisions = precision_score(group_gt, group_pred, average=None, zero_division=0)
#     recalls = recall_score(group_gt, group_pred, average=None, zero_division=0)

#     return {
#         f"{cls}_f1": f1 for cls, f1 in zip(supercategories, f1s)
#     } | {
#         f"{cls}_precision": p for cls, p in zip(supercategories, precisions)
#     } | {
#         f"{cls}_recall": r for cls, r in zip(supercategories, recalls)
#     }

def get_class_combination(row):
    labels = [cat for cat in supercategories if row[cat] == 1]
    return "+".join(labels) if labels else "None"


# Load the ground truth of the test set
path = "/path/to/test.csv"
df = pd.read_csv(path)
df["plot"] = df["filename"].apply(lambda x: x.split("_")[1])
# df["date"] = df["filename"].apply(lambda x: x.split("_")[0].split("-")[-1])
df["month"] = df["filename"].apply(lambda x: x.split("_")[0].split("-")[-1]).str[2:4]
df[supercategories] = df[supercategories].astype(int)
print(df.head())
print(df.shape)

print("Number of plots: ", len(df["plot"].unique()))
print(df["plot"].value_counts())
print("\nNumber of different months: ", len(df["month"].unique()))
print(df["month"].value_counts())

results_dir = "/path/to/trained/model/_test"
# Use only the relevant target columns for calculating the performance
df_gt = df[supercategories]

# Load the predictions
df_pred = pd.read_csv(os.path.join(results_dir, "test_results.csv"))
df_pred["plot"] = df["plot"]
df_pred["month"] = df["month"]
df_pred[supercategories] = (df_pred[supercategories] >= 0.5).astype(int)
# print(df_pred)
print("\n\nSample output:")
print(df.iloc[0])
print(df_pred.iloc[0])
print("\n\n")


## BY PLOT

# Calculate f1-macro by plot
f1_by_plot = df.groupby("plot").apply(
    lambda gp: f1_for_group(
        gp[supercategories], 
        df_pred[df_pred["plot"] == gp.name][supercategories])
    ).reset_index(name="f1_by_plot")
# Per-class f1s by plot
f1_per_class_plot = df.groupby("plot").apply(
    lambda gp: f1_for_group_all_classes(
        df_pred[df_pred["plot"] == gp.name][supercategories],
        gp[supercategories]
    )
).apply(pd.Series).reset_index()
# Calculate number of samples by plot
plot_counts = df.groupby("plot").size().reset_index(name="n_samples")
# Merge and adjust the dataframes
f1_by_plot = f1_by_plot.merge(f1_per_class_plot, on="plot")
f1_by_plot = f1_by_plot.merge(plot_counts, on="plot")
f1_by_plot = f1_by_plot.sort_values(by="f1_by_plot", ascending=False)
f1_by_plot.to_csv("analysis/f1_by_plot.csv", index=False)
print(f1_by_plot)


## BY MONTH

# Calculate f1-macro by month
f1_by_month = df.groupby("month").apply(
    lambda gp: f1_for_group(
        gp[supercategories], 
        df_pred[df_pred["month"] == gp.name][supercategories])
    ).reset_index(name="f1_by_month"
)
# Per-class f1s by month
f1_per_class_month = df.groupby("month").apply(
    lambda gp: f1_for_group_all_classes(
        df_pred[df_pred["month"] == gp.name][supercategories],
        gp[supercategories]
    )
).apply(pd.Series).reset_index()
# Calculate number of samples by month
month_counts = df.groupby("month").size().reset_index(name="n_samples")
# Merge and adjust the dataframes
f1_by_month = f1_by_month.merge(f1_per_class_month, on="month")
f1_by_month = f1_by_month.merge(month_counts, on="month")
f1_by_month = f1_by_month.sort_values(by="f1_by_month", ascending=False)
f1_by_month.to_csv("analysis/f1_by_month.csv", index=False)
print(f1_by_month)


## BY COMBO

# Add combination column to df
df["class_combo"] = df[supercategories].apply(get_class_combination, axis=1)
df_pred["class_combo"] = df["class_combo"] 

# # F1 per combination (macro across all classes)
# f1_by_combo = df.groupby("class_combo").apply(
#     lambda gp: f1_for_group(
#         df_pred[df_pred["class_combo"] == gp.name][supercategories],
#         gp[supercategories]
#     )
# ).reset_index(name="f1_macro")

grouped = df.groupby("class_combo")
results = []

for combo, group_gt in grouped:
    group_pred = df_pred.loc[group_gt.index]

    # Metrics storage
    row = {"class_combo": combo}

    # Per-class scores
    f1s = f1_score(group_gt[supercategories], group_pred[supercategories], average=None, zero_division=0)
    precs = precision_score(group_gt[supercategories], group_pred[supercategories], average=None, zero_division=0)
    recs = recall_score(group_gt[supercategories], group_pred[supercategories], average=None, zero_division=0)

    for i, cls in enumerate(supercategories):
        row[f"{cls}_f1"] = f1s[i]
        row[f"{cls}_precision"] = precs[i]
        row[f"{cls}_recall"] = recs[i]

    # Get only active classes in this combo
    active_classes = combo.split("+") if combo != "None" else []
    if active_classes:
        # Macro F1 over active classes only
        row["f1_macro_active"] = np.mean([row[f"{cls}_f1"] for cls in active_classes])
    else:
        row["f1_macro_active"] = 0.0  # No class active

    # Subset accuracy (exact match)
    exact_match = (group_pred[supercategories].values == group_gt[supercategories].values).all(axis=1).mean()
    row["exact_match_accuracy"] = exact_match

    # Number of samples in group
    row["n_samples"] = len(group_gt)

    results.append(row)

# Convert to DataFrame
df_combo_metrics = pd.DataFrame(results)
df_combo_metrics = df_combo_metrics.sort_values(by="class_combo", ascending=True)
df_combo_metrics.to_csv("analysis/performance_by_combo.csv", index=False)
print(df_combo_metrics)


# Per-class F1 per combination
# f1_per_class_combo = df.groupby("class_combo").apply(
#     lambda gp: f1_for_group_all_classes(
#         df_pred[df_pred["class_combo"] == gp.name][supercategories],
#         gp[supercategories]
#     )
# ).apply(pd.Series).reset_index()

# class_combo_stats = df.groupby("class_combo").apply(
#     lambda gp: metrics_for_group_all_classes(
#         df_pred.loc[gp.index, supercategories],
#         gp[supercategories]
#     )
# ).apply(pd.Series).reset_index()

# class_combo_stats = f1_by_combo.merge(class_combo_stats, on="class_combo")

# class_combo_counts = df.groupby("class_combo").size().reset_index(name="n_samples")
# results_df = class_combo_stats.merge(class_combo_counts, on="class_combo")
# # results_df["f1_macro"] = results_df[[f"{cls}_f1" for cls in supercategories]].mean(axis=1)
# results_df = results_df.sort_values(by="f1_macro", ascending=False)
# print(results_df)

# # Sample counts
# combo_counts = df["class_combo"].value_counts().reset_index()
# combo_counts.columns = ["class_combo", "n_samples"]

# # Merge
# f1_by_combo = f1_by_combo.merge(f1_per_class_combo, on="class_combo")
# f1_by_combo = f1_by_combo.merge(combo_counts, on="class_combo")
# f1_by_combo = f1_by_combo.sort_values(by="f1_macro", ascending=False)
# print(f1_by_combo)

print(f1_by_plot["n_samples"].sum())
print(f1_by_month["n_samples"].sum())
print(df_combo_metrics["n_samples"].sum())



### PLOTTING

# BY PLOT

# Bin the f1 scores for plotting
bins = np.arange(0, 1.1, 0.1)  # 0.0 to 1.0 in steps of 0.1
f1_by_plot["f1_bin"] = pd.cut(f1_by_plot["f1_by_plot"], bins=bins, include_lowest=True)

plt.figure(figsize=(10, 6))
sns.countplot(data=f1_by_plot, x="f1_bin", order=sorted(f1_by_plot["f1_bin"].unique()))
plt.xlabel("F1 Score Range")
plt.ylabel("Number of Plots")
plt.title("Distribution of Plots by Macro F1 Score Bins")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("analysis/plots_performance.png")
plt.close()


# BY MONTH

# Bin the f1 scores for plotting
bins = np.arange(0, 1.1, 0.1)  # 0.0 to 1.0 in steps of 0.1
f1_by_month["f1_bin"] = pd.cut(f1_by_month["f1_by_month"], bins=bins, include_lowest=True)

# Plot as histogram
plt.figure(figsize=(10, 6))
sns.countplot(data=f1_by_month, x="f1_bin", order=sorted(f1_by_month["f1_bin"].unique()))
plt.xlabel("F1 Score Range")
plt.ylabel("Number of months")
plt.title("Distribution of months by Macro F1 Score Bins")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("analysis/months_performance.png")
plt.close()

# Plot for every month in a barplot
plt.figure(figsize=(12, 6))
sns.barplot(data=f1_by_month, x="month", y="f1_by_month", color="steelblue")

plt.xlabel("Month")
plt.ylabel("Macro F1 Score")
plt.title("Average Macro F1 Score by Month")
plt.xticks(rotation=45)
plt.ylim(0, 1.0)  # Optional: to keep y-axis consistent from 0 to 1
plt.tight_layout()
plt.savefig("analysis/f1_by_month_barplot.png")
plt.close()


# BY COMBOS

# Melt the DataFrame to long format for seaborn
f1_long = df_combo_metrics.melt(id_vars="class_combo", 
                                value_vars=["Anth_f1", "Bio_f1", "Geo_f1", "Sil_f1"],
                                var_name="Class", value_name="F1 Score")

# Clean class names
f1_long["Class"] = f1_long["Class"].str.replace("_f1", "")

plt.figure(figsize=(14, 6))
sns.barplot(data=f1_long, x="class_combo", y="F1 Score", hue="Class")
plt.xticks(rotation=45, ha="right")
plt.title("F1 Score by Class Combination")
plt.tight_layout()
plt.savefig("analysis/combos_grouped_bar_chart.png")
plt.close()


# Heatmap - metrics per class
# Set class F1 columns
f1_cols = ["Anth_f1", "Bio_f1", "Geo_f1", "Sil_f1"]
df_f1 = df_combo_metrics.set_index("class_combo")[f1_cols]

plt.figure(figsize=(10, 6))
sns.heatmap(df_f1, annot=True, cmap="Blues", vmin=0, vmax=1)
plt.title("F1 Score Heatmap per Class Combination")
plt.ylabel("Class Combination")
plt.xlabel("Class")
plt.tight_layout()
plt.savefig("analysis/combos_heatmap.png")
plt.close()


# Exact match vs f1 score
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_combo_metrics, 
                x="f1_macro_active", y="exact_match_accuracy", 
                hue="class_combo", s=100)

for i, row in df_combo_metrics.iterrows():
    plt.text(row["f1_macro_active"] + 0.002, row["exact_match_accuracy"] + 0.002, 
             row["class_combo"], fontsize=9)

plt.xlabel("Macro F1 (Active Classes)")
plt.ylabel("Exact Match Accuracy")
plt.title("F1 Macro vs Exact Accuracy by Class Combination")
plt.grid(True)
plt.tight_layout()
plt.savefig("analysis/combos_exact_vs_f1_performance.png")
plt.close()