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
df = pd.read_csv("./analysis/processed_annotations_percentages/adapted_annotations_best_thresholds_paperVersion.csv").set_index("filename")
print(df.head())
print(df.shape)

aggregation_type = "max"

version = "withSilence"
dataset = "pam-data"
# step_variant = "small"
step_variant = "big"

if step_variant == "big":
    predictions_path = f"/path/to/BESound_predictions_dataset_variations/balanced/{version}/{dataset}/results.csv"
else:
    predictions_path = f"/path/to/BESound_predictions_dataset_variations_smallSteps/balanced/{version}/{dataset}/results.csv"
pred = pd.read_csv(predictions_path).set_index("filename")
pred = pred.drop(columns=["prediction", "output"])
if "offset" in pred.columns:
    pred = pred[~pred["offset"].str.contains("majority", case=False, na=False)]
    pred.drop(columns=["offset"], inplace=True)
    if aggregation_type == "max":
        pred = pred.groupby("filename").max()
    else:  
        pred = pred.groupby("filename").mean()
print(pred.head(5))
print(pred.shape)

# Copy max probs for evaluation later (below)
max_probs_df = pred.copy()
# Align indices of predictions and annotations
common_index = max_probs_df.index.intersection(df.index)    
df = df.loc[common_index].sort_index()
max_probs_df = max_probs_df.loc[common_index].sort_index()
max_probs_df = max_probs_df.rename(columns={col: f"{col}_max_prob" for col in max_probs_df.columns})
print("\nMax probs stats:")
print(df.shape)
print(max_probs_df.shape)

# Attention!! 
# Uses the percentage annotations based on the individual class thresholds with the global 0.5 threshold.
# Therefore, the chosen duration percentage per class is not necessarily the best for the 0.5 threshold as well...
# The evaluation for the best percentage together with the 0.5 thresholds is now done in evaluate_percentile_annotations.py

# print("\n--- Adapted annotations with 0.5 ---")

# # Convert the predictions to integers based on predefined thresholds
# thresholds = {
#     "Anth": 0.5,
#     "Bio": 0.5,
#     "Geo": 0.5,
#     "Sil": 0.9
# }
# pred = pred.apply(lambda col: (col > thresholds[col.name]).astype(int))

# # Align indices of predictions and annotations
# common_index = pred.index.intersection(df.index)    
# df = df.loc[common_index].sort_index()
# pred = pred.loc[common_index].sort_index()
# print("df and pred shapes:")
# print(df.shape)
# print(pred.shape)


# print("Class -- Precision -- Recall -- F1-Score")
# for col in target_classes:
#     prec = precision_score(df[col], pred[col])
#     rec = recall_score(df[col], pred[col])
#     f1 = f1_score(df[col], pred[col])
#     print(f"{col} -- {prec:.3f} -- {rec:.3f} -- {f1:.3f}")
# print("Total F1-macro: ", f1_score(df[target_classes], pred[target_classes], average="macro"))




###############################


def best_threshold_by_f1(y_true, y_score):
    """
    Returns the threshold in [0,1] that maximizes F1 on y_true given continuous scores y_score.
    Uses precision_recall_curve; ties broken by higher recall.
    Handles edge cases with no positives/negatives.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    # If all labels are the same, fall back to 0.5
    if y_true.max() == 0:
        return 1.0  # predict all negatives
    if y_true.min() == 1:
        return 0.0  # predict all positives

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns len(thresholds) = len(precisions) - 1
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    # Compute F1 per threshold; avoid division by zero
    denom = (precisions + recalls)
    f1s = np.where(denom > 0, 2 * precisions * recalls / denom, 0.0)

    # Choose the threshold with max F1; break ties by higher recall, then lower threshold
    best_idx = np.lexsort((
        thresholds,              # prefer lower threshold among ties
        -recalls,                # then higher recall
        -f1s                     # primary: higher F1
    ))[0]

    lexsort = np.lexsort((
        thresholds,              # prefer lower threshold among ties
        -recalls,                # then higher recall
        -f1s                     # primary: higher F1
    ))
    best_idx = lexsort[0]
    print("\nLexsort: ", lexsort)
    print("\nBest idx: ", best_idx)

    return float(thresholds[best_idx]) if thresholds.size else 0.5

print("\n --- Adapted annotations with individual thresholds ---")

target_classes = ["Anth", "Bio", "Geo"]

# df_base = pd.read_csv("./analysis/processed_annotations_percentiles/baseline_annotations.csv").set_index("filename")
# df_p5 = pd.read_csv("./analysis/processed_annotations_percentiles/adapted_annotations_p5.csv").set_index("filename")
# df_p10 = pd.read_csv("./analysis/processed_annotations_percentiles/adapted_annotations_p10.csv").set_index("filename")
# df_p25 = pd.read_csv("./analysis/processed_annotations_percentiles/adapted_annotations_p25.csv").set_index("filename")
df = pd.read_csv("./analysis/processed_annotations_percentages/adapted_annotations_best_thresholds_paperVersion.csv").set_index("filename")
print(df.head())
print(df.shape)

aggregation_type = "max"

version = "withSilence"
dataset = "pam-data"
# step_variant = "small"
step_variant = "big"

if step_variant == "big":
    predictions_path = f"/path/to/CoarseSoundNet/BESound_predictions_dataset_variations/balanced/{version}/{dataset}/results.csv"
else:
    predictions_path = f"/path/to/CoarseSoundNet/BESound_predictions_dataset_variations_smallSteps/balanced/{version}/{dataset}/results.csv"
pred = pd.read_csv(predictions_path).set_index("filename")
pred = pred.drop(columns=["prediction", "output"])
if "offset" in pred.columns:
    pred = pred[~pred["offset"].str.contains("majority", case=False, na=False)]
    pred.drop(columns=["offset"], inplace=True)
    if aggregation_type == "max":
        pred = pred.groupby("filename").max()
    else:  
        pred = pred.groupby("filename").mean()
print(pred.head(5))
print(pred.shape)

# Align indices of predictions and annotations
common_index = pred.index.intersection(df.index)    
df = df.loc[common_index].sort_index()
pred = pred.loc[common_index].sort_index()
print("df and pred shapes:")
print(df.shape)
print(pred.shape)

# Learn thresholds per class
learned_thresholds = {}
for cls in target_classes:
    y_true = df[cls].values
    y_score = pred[cls].values
    t = best_threshold_by_f1(y_true, y_score)
    learned_thresholds[cls] = round(t, 4)

# You can keep a fixed threshold for classes not in target_classes (e.g., "Sil")
# Merge with any defaults you want to keep
defaults = {"Sil": 0.9}
thresholds = {**defaults, **learned_thresholds}

print("Learned thresholds:")
for k, v in thresholds.items():
    print(f"  {k}: {v}")

# Apply thresholds to binarize predictions
pred_bin = pred.copy()
for cls in target_classes:
    pred_bin[cls] = (pred[cls].values > thresholds[cls]) #.astype(int)
# If you keep "Sil" elsewhere, only binarize if present
# if "Sil" in pred_bin.columns and "Sil" in thresholds:
    # pred_bin["Sil"] = (pred["Sil"].values > thresholds["Sil"]).astype(int)
if "Sil" in pred_bin.columns:
    pred_bin["Sil"] = ~(pred_bin["Anth"] | pred_bin["Bio"] | pred_bin["Geo"])
pred_bin = pred_bin.astype(int)
print(pred_bin.head())
print(pred_bin.shape)


# Evaluate per-class and macro-F1
print("\nClass -- Precision -- Recall -- F1-Score")
for cls in target_classes:
    prec = precision_score(df[cls], pred_bin[cls], zero_division=0)
    rec  = recall_score(df[cls], pred_bin[cls], zero_division=0)
    f1   = f1_score(df[cls], pred_bin[cls], zero_division=0)
    print(f"{cls} -- {prec:.3f} -- {rec:.3f} -- {f1:.3f}")

macro_f1 = f1_score(df[target_classes], pred_bin[target_classes], average="macro", zero_division=0)
print("Total F1-macro:", round(macro_f1, 3))


# Combine df and pre to combined_df for further evaluations later (see below)
df = df.rename(columns={col: f"gt_new_{col}" for col in df.columns})
pred_bin = pred_bin.rename(columns={col: f"pred_new_{col}" for col in pred_bin.columns})
df_combined = pd.concat([max_probs_df, df, pred_bin], axis=1)

# Load indices and duration csv
summary_df = pd.read_csv("/path/to/BESound_indices_summary.csv").set_index("filename")
summary_df = summary_df.iloc[:, :10]

# Align indices of predictions and annotations
common_index = summary_df.index.intersection(df_combined.index)    
df_combined = df_combined.loc[common_index].sort_index()
summary_df = summary_df.loc[common_index].sort_index()

# Newly composited summary dataframe
summary_df = pd.concat([summary_df, df_combined], axis=1).reset_index()
summary_df.to_csv("/path/to/BESound_indices_summary_paperVersion.csv", index=False)


##############################################
print("\n############################################################\n")
print("\n--- Adapted annotations with special thresholds and counts ---")

version = "withSilence"
dataset = "pam"
step_variant = "small"
# step_variant = "big"

df = pd.read_csv("./analysis/processed_annotations_percentages/adapted_annotations_best_thresholds_paperVersion.csv").set_index("filename")
predictions_path = f"/path/to/BESound_predictions_dataset_variations_{step_variant}Steps/balanced/{version}/{dataset}/results.csv"

# Evaluating based on specialised and adpated thresholds
threshold = .4
n_windows = 51 # for ws=10 and stepsize=1
thresholds = {"Anth": learned_thresholds["Anth"], "Bio": learned_thresholds["Bio"], "Geo": learned_thresholds["Geo"]}
# thresholds = {"Anth": 0.5, "Bio": 0.5, "Geo": 0.5}
percentages = [.05, .10, .15, .20, .25]


for p in percentages:
    count = int(p * n_windows)
    print(f"P = {p} => count = {count}")

    # Load and prepare predictions for new thresholds
    pred = pd.read_csv(predictions_path).set_index("filename")
    pred = pred.groupby("filename").apply(
        lambda x: pd.Series({
            "Anth": (x["Anth"] > thresholds["Anth"]).sum() >= count,
            "Bio": (x["Bio"] > thresholds["Bio"]).sum() >= count, 
            "Geo": (x["Geo"] > thresholds["Geo"]).sum() >= count,
            # "Sil": (x["Sil"] > threshold).sum() >= 55, # 55
        })
    )
    pred["Sil"] = pred.apply(lambda x: ~x["Anth"] & ~x["Bio"] & ~x["Geo"], axis=1)


    # Align indices of predictions and annotations
    common_index = pred.index.intersection(df.index)    
    df = df.loc[common_index].sort_index()
    pred = pred.loc[common_index].sort_index()
    # print("df and pred shapes:")
    # print(df.shape)
    # print(pred.shape)


    ### EVALUATION
    print("Col -- Prec -- Rec -- F1")
    for col in target_classes:
        # On the whole test set
        prec = precision_score(df[col], pred[col])
        rec = recall_score(df[col], pred[col])
        f1 = f1_score(df[col], pred[col])
        print(f"{col} -- {prec:.3f} -- {rec:.3f} -- {f1:.3f}")
    print("Total F1-macro: ", f1_score(df[target_classes], pred[target_classes], average="macro"))


    print(f"\nTotal macro F1-score:")
    print(f1_score(df[target_classes], pred[target_classes], average="macro"))


# Calculate the count-based approach with the best count thresholds

df = pd.read_csv("./analysis/processed_annotations_percentages/adapted_annotations_best_thresholds_paperVersion.csv").set_index("filename")
predictions_path = f"/path/to/CoarseSoundNet/BESound_predictions_dataset_variations_smallSteps/balanced/{version}/{dataset}/results.csv"

# Evaluating based on specialised and adpated thresholds
thresholds = {"Anth": learned_thresholds["Anth"], "Bio": learned_thresholds["Bio"], "Geo": learned_thresholds["Geo"]}

# Load and prepare predictions for new thresholds
pred = pd.read_csv(predictions_path).set_index("filename")
pred = pred.groupby("filename").apply(
    lambda x: pd.Series({
        "Anth": (x["Anth"] > thresholds["Anth"]).sum() >= 2,
        "Bio": (x["Bio"] > thresholds["Bio"]).sum() >= 5, 
        "Geo": (x["Geo"] > thresholds["Geo"]).sum() >= 10,
        # "Sil": (x["Sil"] > threshold).sum() >= 55, # 55
    })
)
pred["Sil"] = pred.apply(lambda x: ~x["Anth"] & ~x["Bio"] & ~x["Geo"], axis=1)


# Align indices of predictions and annotations
common_index = pred.index.intersection(df.index)    
df = df.loc[common_index].sort_index()
pred = pred.loc[common_index].sort_index()

### EVALUATION
print("\nBest counts per class:")
print("Col -- Prec -- Rec -- F1")
for col in target_classes:
    # On the whole test set
    prec = precision_score(df[col], pred[col])
    rec = recall_score(df[col], pred[col])
    f1 = f1_score(df[col], pred[col])
    print(f"{col} -- {prec:.3f} -- {rec:.3f} -- {f1:.3f}")
print("Total F1-macro: ", f1_score(df[target_classes], pred[target_classes], average="macro"))
