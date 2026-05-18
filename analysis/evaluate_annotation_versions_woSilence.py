import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def evaluate_metrics(y_true, y_pred, labels=["Anth", "Bio", "Geo"]):
    f1s = []
    for col in labels:
        f1 = f1_score(y_true[col], y_pred[col])
        f1s.append(f1)
    return f1s


def bootstrap_ci(y_true, y_pred, n_bootstraps=1000, ci=95, labels=["Anth", "Bio", "Geo"]):
    np.random.seed(42)
    n_samples = len(y_true)
    stats = {label: [] for label in labels}

    for _ in range(n_bootstraps):
        sample_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_sample = y_true.iloc[sample_idx]
        y_pred_sample = y_pred.iloc[sample_idx]
        f1s = evaluate_metrics(y_true_sample, y_pred_sample, labels)
        for label, f1 in zip(labels, f1s):
            stats[label].append(f1)

    ci_low = (100 - ci) / 2
    ci_high = 100 - ci_low
    ci_bounds = {}

    for label in labels:
        lower = np.percentile(stats[label], ci_low)
        upper = np.percentile(stats[label], ci_high)
        ci_bounds[label] = (lower, upper)

    return ci_bounds


def bootstrap_macro_ci(y_true, y_pred, n_bootstraps=1000, ci=95, labels=["Bio", "Geo", "Sil"]):
    np.random.seed(42)
    n_samples = len(y_true)
    macro_f1_scores = []

    for _ in range(n_bootstraps):
        sample_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_sample = y_true.iloc[sample_idx]
        y_pred_sample = y_pred.iloc[sample_idx]

        # Compute macro F1 across all labels
        f1 = f1_score(y_true_sample[labels], y_pred_sample[labels], average='macro')
        macro_f1_scores.append(f1)

    # Compute CI bounds
    ci_low = (100 - ci) / 2
    ci_high = 100 - ci_low
    lower = np.percentile(macro_f1_scores, ci_low)
    upper = np.percentile(macro_f1_scores, ci_high)

    return lower, upper



# Settings 
compute_CIs = False

# Get the newly annotated filenames
df_newly_annotated = pd.read_csv("./analysis/processed_annotations/new_annotations_v1.csv").set_index("filename")
newly_annotated_files = df_newly_annotated.index.tolist()

# Load annotations
df_max_prob = pd.read_csv("./analysis/processed_annotations/old_annotations.csv").set_index("filename")
# df_max_prob = pd.read_csv("./analysis/processed_annotations_percentages/baseline_annotations.csv").set_index("filename")
print(df_max_prob.head(), df_max_prob.shape)
df_adapted_thresholds = pd.read_csv("./analysis/processed_annotations/adapted_th_annotations.csv").set_index("filename")
df_adapted_thresholds_v1 = pd.read_csv("./analysis/processed_annotations/adapted_th_annotations_v1.csv").set_index("filename")
print(df_adapted_thresholds_v1.head(), df_adapted_thresholds_v1.shape)
df_adapted_thresholds_v2 = pd.read_csv("./analysis/processed_annotations/adapted_th_annotations_v2.csv").set_index("filename")



# Evaluating with the max probability
version = "hf"
# version="edansa_svenja_mix"
# version="edansa_hf"
if version == "hf":
    predictions_path = "/path/to/predictions/reviewed_annotations/edansa_hf_10/results.csv"
else:
    predictions_path = f"/path/to/predictions/reviewed_annotations/{version}/results.csv"


version = "withSilence"
model = "clap"
dataset = "pam-data"
predictions_path = f"/predictions/{version}/{dataset}/results.csv"

pred = pd.read_csv(predictions_path).set_index("filename")
pred = pred.drop(columns=["prediction", "output"])

pred = pred.drop(columns=["Sil"])

# individual thresholds
thresholds = {
    "Anth": 0.5,
    "Bio": 0.5,
    "Geo": 0.5,
    # "Sil": 0.9
}

target_columns = ["Anth", "Bio", "Geo"]

if "offset" in pred.columns:
    pred.drop(columns=["offset"], inplace=True)
pred = pred.groupby("filename").max()

# Make a copy for probs
probs = pred.copy()
probs = probs.rename(columns={col: f"{col}_max_prob" for col in probs.columns if col != probs.index.name})
print(probs.shape)

pred = pred.apply(lambda col: (col > thresholds[col.name]).astype(int))
# pred["Sil"] = pred.apply(lambda x: int((x["Sil"] & (x.sum() == 1)) | (x.sum() == 0)), axis=1) # adjusted operator logic with brackets
# Rename for combined analysis
pred_tmp = pred.rename(columns={col: f"{col}_max_pred" for col in pred.columns if col != pred.index.name})
print(pred.shape)

# Align indices for the combined csv
common_index = probs.index.intersection(pred.index)
probs = probs.loc[common_index]
probs = probs.sort_index()
pred_tmp = pred_tmp.loc[common_index]
pred_tmp = pred_tmp.sort_index()

df_max_prob_tmp = df_max_prob.copy()
df_max_prob_tmp = df_max_prob.rename(columns={col: f"gt_old_{col}" for col in df_max_prob.columns if col != df_max_prob.index.name})

combined_df = pd.concat([probs, pred_tmp, df_max_prob_tmp], axis=1)
common_index = df_max_prob.index.intersection(combined_df.index)
combined_df = combined_df.loc[common_index]
combined_df = combined_df.sort_index()
combined_df = combined_df.reset_index()
# combined_df.to_csv(predictions_path.replace("results", "max_probs"), index=False)
print(combined_df.shape)
print(combined_df.head())

# Align indices
common_index = df_max_prob.index.intersection(pred.index)
pred = pred.loc[common_index]
pred = pred.sort_index()

### EVALUATION

print("\nVersion: Max Prob")
print("\nWhole test set:")
print("Col -- Prec -- Rec -- F1")
for col in ["Anth", "Bio", "Geo"]:
    prec = precision_score(df_max_prob[col], pred[col])
    rec = recall_score(df_max_prob[col], pred[col])
    f1 = f1_score(df_max_prob[col], pred[col])
    print(f"{col} -- {prec:.2f} -- {rec:.2f} -- {f1:.2f}")
print("Total F1-macro: ", f1_score(df_max_prob[target_columns], pred[target_columns], average="macro"))

print("\nReviewed test files:")
print("Col -- Prec -- Rec -- F1")
for col in ["Anth", "Bio", "Geo"]:
    # Only on the reviewed annotations
    prec = precision_score(df_max_prob.loc[newly_annotated_files][col], pred.loc[newly_annotated_files][col])
    rec = recall_score(df_max_prob.loc[newly_annotated_files][col], pred.loc[newly_annotated_files][col])
    f1 = f1_score(df_max_prob.loc[newly_annotated_files][col], pred.loc[newly_annotated_files][col])
    print(f"{col} -- {prec:.2f} -- {rec:.2f} -- {f1:.2f}")

# Initialize the DataFrame for storing F1 Scores
f1_data = {}

# Evaluating with the max probability 
f1_data[("Max Prob", "Whole test set")] = [
    f1_score(df_max_prob["Anth"], pred["Anth"]),
    f1_score(df_max_prob["Bio"], pred["Bio"]),
    f1_score(df_max_prob["Geo"], pred["Geo"]),
    # f1_score(df_max_prob["Sil"], pred["Sil"])
]

f1_data[("Max Prob", "Subset")] = [
    f1_score(df_max_prob.loc[newly_annotated_files]["Anth"], pred.loc[newly_annotated_files]["Anth"]),
    f1_score(df_max_prob.loc[newly_annotated_files]["Bio"], pred.loc[newly_annotated_files]["Bio"]),
    f1_score(df_max_prob.loc[newly_annotated_files]["Geo"], pred.loc[newly_annotated_files]["Geo"]),
    # f1_score(df_max_prob.loc[newly_annotated_files]["Sil"], pred.loc[newly_annotated_files]["Sil"])
]



# Evaluating based on specialised and adpated thresholds
threshold = .4

# Load and prepare predictions for new thresholds
pred = pd.read_csv(predictions_path).set_index("filename")
pred = pred.groupby("filename").apply(
    lambda x: pd.Series({
        "Anth": (x["Anth"] > .5).sum() >= 1, # & ((x["Bio"] > .1).sum() <= 35) & ((x["Geo"] > .1).sum() <= 15), #((x["Anth"] > .1).sum() >= 25) & ((x["Bio"] > .1).sum() <= 35) & ((x["Geo"] > .1).sum() <= 15)
        "Bio": (x["Bio"] > .5).sum() >= 1, # 45
        "Geo": (x["Geo"] > .5).sum() >= 1, # 45
        # "Sil": (x["Sil"] > threshold).sum() >= 55, # 55
    })
)
# pred["Sil"] = pred.apply(lambda x: ~x["Anth"] & ~x["Bio"] & ~x["Geo"], axis=1)

# print(pred.head())
# print(pred["Sil"].value_counts())
# print("\n\n")

# Align indices
common_index = df_adapted_thresholds.index.intersection(pred.index)
pred = pred.loc[common_index]
pred = pred.sort_index()


### EVALUATION

for idx, df in enumerate([df_adapted_thresholds, df_adapted_thresholds_v1, df_adapted_thresholds_v2]):
    print("\nVersion: ", idx)
    print("\nWhole test set:")
    print("Col -- Prec -- Rec -- F1")
    for col in ["Anth", "Bio", "Geo"]:
        # On the whole test set
        prec = precision_score(df[col], pred[col])
        rec = recall_score(df[col], pred[col])
        f1 = f1_score(df[col], pred[col])
        print(f"{col} -- {prec:.2f} -- {rec:.2f} -- {f1:.2f}")
    print("Total F1-macro: ", f1_score(df[target_columns], pred[target_columns], average="macro"))

    print("\nReviewed test files:")
    print("Col -- Prec -- Rec -- F1")
    for col in ["Anth", "Bio", "Geo"]:
        # Only on the reviewed annotations
        prec = precision_score(df.loc[newly_annotated_files][col], pred.loc[newly_annotated_files][col])
        rec = recall_score(df.loc[newly_annotated_files][col], pred.loc[newly_annotated_files][col])
        f1 = f1_score(df.loc[newly_annotated_files][col], pred.loc[newly_annotated_files][col])
        print(f"{col} -- {prec:.2f} -- {rec:.2f} -- {f1:.2f}")


# Evaluating each version with f1-score
for idx, (version_name, df) in enumerate([
    ("Adapted Thresholds (AT)", df_adapted_thresholds),
    ("AT - Version 1", df_adapted_thresholds_v1),
    ("AT - Version 2", df_adapted_thresholds_v2)
]):
    if version_name == "AT - Version 1":
        df_tmp = df.sort_index()
        df_tmp = df_tmp.rename(columns={col: f"gt_new_{col}" for col in df_tmp.columns if col != df_tmp.index.name})
        pred_tmp = pred.sort_index().astype(int)
        pred_tmp = pred_tmp.rename(columns={col: f"pred_new_{col}" for col in pred_tmp.columns if col != pred_tmp.index.name})
        new_version_df = pd.concat([df_tmp, pred_tmp], axis=1)
        # print(new_version_df.head(), new_version_df.shape)

        combined_df = combined_df.set_index("filename")
        # print(combined_df.head())

        combined_df = pd.concat([combined_df, new_version_df], axis=1)
        combined_df = combined_df.reset_index()
        print(combined_df.head(), combined_df.shape)
        # combined_df.to_csv(predictions_path.replace("results", "combined_pred_prob_gt"), index=False)
        print("Saved combined probs-preds-gts as csv.")
    

    f1_data[(version_name, "Whole test set")] = [
        f1_score(df["Anth"], pred["Anth"]),
        f1_score(df["Bio"], pred["Bio"]),
        f1_score(df["Geo"], pred["Geo"]),
        # f1_score(df["Sil"], pred["Sil"])
    ]

    # Only on the reviewed annotations
    f1_data[(version_name, "Subset")] = [
        f1_score(df.loc[newly_annotated_files]["Anth"], pred.loc[newly_annotated_files]["Anth"]),
        f1_score(df.loc[newly_annotated_files]["Bio"], pred.loc[newly_annotated_files]["Bio"]),
        f1_score(df.loc[newly_annotated_files]["Geo"], pred.loc[newly_annotated_files]["Geo"]),
        # f1_score(df.loc[newly_annotated_files]["Sil"], pred.loc[newly_annotated_files]["Sil"])
    ]

    print(f"\nTotal macro F1-score: {version_name}")
    print("Whole test set: ", f1_score(df[target_columns], pred[target_columns], average="macro"))
    print("subset: ", f1_score(df.loc[newly_annotated_files][target_columns], pred.loc[newly_annotated_files][target_columns], average="macro"))
    print(f"Total macro Recall: {version_name}")
    print("Whole test set: ", recall_score(df[target_columns], pred[target_columns], average="macro"))
    print("subset: ", recall_score(df.loc[newly_annotated_files][target_columns], pred.loc[newly_annotated_files][target_columns], average="macro"))
    print(f"Total macro Precision: {version_name}")
    print("Whole test set: ", precision_score(df[target_columns], pred[target_columns], average="macro"))
    print("subset: ", precision_score(df.loc[newly_annotated_files][target_columns], pred.loc[newly_annotated_files][target_columns], average="macro"))

    # Compute confidence intervals for each class, if set to true
    if compute_CIs:
        # Class-wise
        ci = bootstrap_ci(df, pred)  # Whole test set
        print("\n95% Confidence Intervals for F1 scores (Whole test set):")
        for label in ["Anth", "Bio", "Geo"]:
            print(f"{label}: {ci[label][0]:.2f} - {ci[label][1]:.2f}")
        
        # Across classes
        ci_low, ci_high = bootstrap_macro_ci(df, pred)
        print(f"95% CI for macro F1-score: {ci_low:.3f} – {ci_high:.3f}")


# Create the DataFrame
df_f1 = pd.DataFrame(f1_data, index=["Anth", "Bio", "Geo"]).T


# Save to CSV
# df_f1.to_csv(predictions_path.replace("results", f'f1_score_summary_{version}'))

# Display the DataFrame
print(df_f1)