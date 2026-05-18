import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve, average_precision_score

# target_classes = ["Anth", "Bio", "Geo", "Sil"]
target_classes = ["Anth", "Bio", "Geo"]

# df_base = pd.read_csv("./analysis/processed_annotations_percentiles/baseline_annotations.csv").set_index("filename")
# df_p5 = pd.read_csv("./analysis/processed_annotations_percentiles/adapted_annotations_p5.csv").set_index("filename")
# df_p10 = pd.read_csv("./analysis/processed_annotations_percentiles/adapted_annotations_p10.csv").set_index("filename")
# df_p25 = pd.read_csv("./analysis/processed_annotations_percentiles/adapted_annotations_p25.csv").set_index("filename")
df_base = pd.read_csv("./analysis/processed_annotations_percentages/baseline_annotations.csv").set_index("filename")
df_p5 = pd.read_csv("./analysis/processed_annotations_percentages/adapted_annotations_p5.csv").set_index("filename")
df_p10 = pd.read_csv("./analysis/processed_annotations_percentages/adapted_annotations_p10.csv").set_index("filename")
df_p25 = pd.read_csv("./analysis/processed_annotations_percentages/adapted_annotations_p25.csv").set_index("filename")
# print(df_p5.head())
# print(df_p5.shape)

# Dictionary containing all annotation variants
annotations = {
    "baseline": df_base,
    "p5": df_p5,
    "p10": df_p10,
    "p25": df_p25
}

aggregation_type = "max" # max seems to be best for the percentages approachs with window_size=10s and step_size=10s

version = "withSilence"
dataset = "pam-data"
predictions_path = f"/path/to/BESound_predictions_dataset_variations/balanced/{version}/{dataset}/results.csv"
pred = pd.read_csv(predictions_path).set_index("filename")
pred = pred.drop(columns=["prediction", "output"])
if "offset" in pred.columns:
    pred = pred[~pred["offset"].str.contains("majority", case=False, na=False)]
    pred.drop(columns=["offset"], inplace=True)
    if aggregation_type == "max":
        pred = pred.groupby("filename").max()
    else:  
        pred = pred.groupby("filename").mean()
# print(pred.head(10))
# print(pred.shape)


# Align indices 
for variant, df_ann in annotations.items():
    common_index = pred.index.intersection(df_ann.index)    
    annotations[variant] = df_ann.loc[common_index].sort_index()
pred = pred.loc[common_index].sort_index()
# print(pred.shape)

### GET PERCENTAGES FOR ALL THRESHOLDS=0.5 ###
print("\n--- Get best percentages with default 0.5 thresholds ---")
pred_base = pred.copy()
pred_base = pred_base.apply(lambda col: (col > 0.5)) #.astype(int)
pred_base["Sil"] = ~(pred_base["Anth"] | pred_base["Bio"] | pred_base["Geo"])
pred_base = pred_base.astype(int)

for variant in annotations.keys():
    print("\nVariant: ", variant)
    # Evaluate per-class and macro-F1
    print("Class -- Precision -- Recall -- F1-Score")
    for cls in target_classes:
        df_variant = annotations[variant]
        prec = precision_score(df_variant[cls], pred_base[cls], zero_division=0)
        rec  = recall_score(df_variant[cls], pred_base[cls], zero_division=0)
        f1   = f1_score(df_variant[cls], pred_base[cls], zero_division=0)
        print(f"{cls} -- {prec:.3f} -- {rec:.3f} -- {f1:.3f}")

    macro_f1 = f1_score(df_variant[target_classes], pred_base[target_classes], average="macro", zero_division=0)
    print("Total F1-macro:", round(macro_f1, 3))

df_best_p = df_variant[target_classes].copy()
df_best_p["Anth"] = annotations["baseline"]["Anth"]
df_best_p["Bio"] = annotations["baseline"]["Bio"]
df_best_p["Geo"] = annotations["p5"]["Geo"]

print("\nBest percentages:")
for cls in target_classes:
    prec = precision_score(df_best_p[cls], pred_base[cls], zero_division=0)
    rec  = recall_score(df_best_p[cls], pred_base[cls], zero_division=0)
    f1   = f1_score(df_best_p[cls], pred_base[cls], zero_division=0)
    print(f"{cls} -- {prec:.3f} -- {rec:.3f} -- {f1:.3f}")

macro_f1 = f1_score(df_best_p[target_classes], pred_base[target_classes], average="macro", zero_division=0)
print("Total F1-macro:", round(macro_f1, 3))


### ROC Curves
print("\n--- ROC-Curves ---")
# fig, axes = plt.subplots(2, 2, figsize=(12,10))
fig, axes = plt.subplots(1, 3, figsize=(18,6))
axes = axes.flatten()
all_best_thresholds = {}

for i, col in enumerate(target_classes):
    print(f"Class: ", col)
    ax = axes[i]

    best_thresholds = {}
    for name, df_ann in annotations.items():
        y_true = df_ann[col]
        y_pred = pred[col]

        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr) 
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

        # calculate Youden's J
        j_scores = tpr - fpr
        j_best_idx = np.argmax(j_scores)
        print(f"Best Youden's J {name}: ", j_scores[j_best_idx])
        best_thresholds[name] = thresholds[j_best_idx]

        # Mark the best threshold point on the curve
        ax.plot(fpr[j_best_idx], tpr[j_best_idx], "o", color="black")
    
    all_best_thresholds[col] = best_thresholds
    print(best_thresholds)
    
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)")
    ax.set_title(f"ROC Curves for {col}", fontsize=18, pad=8)
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
# plt.savefig(f"./analysis/percentages_ROC-curves_all_classes_{aggregation_type}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"./analysis/percentages_3classes_ROC-curves_all_classes_{aggregation_type}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"./analysis/percentages_3classes_ROC-curves_all_classes_{aggregation_type}.pdf", bbox_inches="tight")
plt.close()

# Print the thresholds nicely
for col, thr_dict in all_best_thresholds.items():
    print(f"\nBest thresholds for {col}:")
    for name, thr in thr_dict.items():
        print(f"  {name}: {thr:.3f}")


### NOW PRECISION-RECALL CURVE
print("\n--- Precision-Recall Curves ---")

# fig, axes = plt.subplots(2, 2, figsize=(12,10))
fig, axes = plt.subplots(1, 3, figsize=(18,6))
axes = axes.flatten()
all_best_thresholds = {}

for i, col in enumerate(target_classes):
    print("\nClass: ", col)
    ax = axes[i]

    best_thresholds = {}
    for name, df_ann in annotations.items():
        y_true = df_ann[col]
        y_pred = pred[col]

        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        # print(precision)
        # print(precision.shape)

        ap = average_precision_score(y_true, y_pred)
        ax.plot(recall, precision, label=f"{name} (AP = {ap:.2f})")
    
        # Youden-like criterion
        f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
        best_idx = np.argmax(f1_scores)
        print(f"Best F1-score {name}: ", f1_scores[best_idx])
        best_thresholds[name] = thresholds[best_idx]

        # Mark the best threshold point
        ax.plot(recall[best_idx], precision[best_idx], "o", color="black")

    all_best_thresholds[col] = best_thresholds
    print(best_thresholds)
    
    ax.set_title(f"Precision-Recall Curves for {col}", fontsize=18, pad=8)
    ax.set_xlabel("Recall", fontsize=16)
    ax.set_ylabel("Precision", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
# plt.savefig(f"./analysis/percentages_PR-curves_all_classes_{aggregation_type}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"./analysis/percentages_3classes_PR-curves_all_classes_{aggregation_type}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"./analysis/percentages_3classes_PR-curves_all_classes_{aggregation_type}.pdf", bbox_inches="tight")
plt.close()

# Print the thresholds nicely
for col, thr_dict in all_best_thresholds.items():
    print(f"\nBest thresholds for {col}:")
    for name, thr in thr_dict.items():
        print(f"  {name}: {thr:.3f}")


################
# Calculate results for best thresholds on the baseline annotations
print("\n--- Calculate with CST on the baseline annotations ---")
pred = pd.read_csv(predictions_path).set_index("filename")
pred = pred.drop(columns=["prediction", "output"])
if "offset" in pred.columns:
    pred = pred[~pred["offset"].str.contains("majority", case=False, na=False)]
    pred.drop(columns=["offset"], inplace=True)
    pred = pred.groupby("filename").max()
pred = pred.drop(columns=["Sil"])

# Align indices for the combined csv
common_index = pred.index.intersection(df_base.index)    
df_tmp = df_base.loc[common_index].sort_index()
pred = pred.loc[common_index].sort_index()

thresholds = {
    "Anth": all_best_thresholds["Anth"]["baseline"],
    "Bio": all_best_thresholds["Bio"]["baseline"],
    "Geo": all_best_thresholds["Geo"]["baseline"],
}
print("Thresholds: ", thresholds)

# Binarize
pred = pred.apply(lambda col: (col > thresholds[col.name]))
pred["Sil"] = ~(pred["Anth"] | pred["Bio"] | pred["Geo"])
pred = pred.astype(int)

print("Class -- Precision -- Recall -- F1-Score")
for cls in target_classes:
    prec = precision_score(df_tmp[cls], pred[cls], zero_division=0)
    rec  = recall_score(df_tmp[cls], pred[cls], zero_division=0)
    f1   = f1_score(df_tmp[cls], pred[cls], zero_division=0)
    print(f"{cls} -- {prec:.3f} -- {rec:.3f} -- {f1:.3f}")

macro_f1 = f1_score(df_tmp[target_classes], pred[target_classes], average="macro", zero_division=0)
print("Total F1-macro:", round(macro_f1, 3))