import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Get the newly annotated filenames
df_newly_annotated = pd.read_csv("./analysis/processed_annotations/new_annotations_v1.csv").set_index("filename")
newly_annotated_files = df_newly_annotated.index.tolist()

# Load annotations
df_max_prob = pd.read_csv("./analysis/processed_annotations/old_annotations.csv").set_index("filename")
df_adapted_thresholds = pd.read_csv("./analysis/processed_annotations/adapted_th_annotations.csv").set_index("filename")
df_adapted_thresholds_v1 = pd.read_csv("./analysis/processed_annotations/adapted_th_annotations_v1.csv").set_index("filename")
df_adapted_thresholds_v2 = pd.read_csv("./analysis/processed_annotations/adapted_th_annotations_v2.csv").set_index("filename")

# Select the annotation version to compare with
df = df_adapted_thresholds_v1.copy()


# Evaluating with the max probability
version = "hf"
# version="edansa_svenja_mix"
if version == "hf":
    predictions_path = "/path/to/predictions/reviewed_annotations/edansa_hf_10/results.csv"
else:
    predictions_path = f"/path/to/predictions/reviewed_annotations/{version}/results.csv"
pred = pd.read_csv(predictions_path).set_index("filename")
pred = pred.drop(columns=["prediction", "output"])

# # individual thresholds
# thresholds = {
#     "Anth": 0.5,
#     "Bio": 0.5,
#     "Geo": 0.5,
#     "Sil": 0.9
# }

# if "offset" in pred.columns:
#     pred.drop(columns=["offset"], inplace=True)
# pred = pred.groupby("filename").max()
# pred = pred.apply(lambda col: (col > thresholds[col.name]).astype(int))
# pred["Sil"] = pred.apply(lambda x: int((x["Sil"] & x.sum() == 1) | (x.sum() == 0)), axis=1)

# # Align indices
# common_index = df_max_prob.index.intersection(pred.index)
# pred = pred.loc[common_index]
# pred = pred.sort_index()

# ### EVALUATION

# print("\nVersion: Max Prob")
# print("\nWhole test set:")
# print("Col -- Prec -- Rec -- F1")
# for col in ["Anth", "Bio", "Geo", "Sil"]:
#     prec = precision_score(df_max_prob[col], pred[col])
#     rec = recall_score(df_max_prob[col], pred[col])
#     f1 = f1_score(df_max_prob[col], pred[col])
#     print(f"{col} -- {prec:.2f} -- {rec:.2f} -- {f1:.2f}")

# print("\nReviewed test files:")
# print("Col -- Prec -- Rec -- F1")
# for col in ["Anth", "Bio", "Geo", "Sil"]:
#     # Only on the reviewed annotations
#     prec = precision_score(df_max_prob.loc[newly_annotated_files][col], pred.loc[newly_annotated_files][col])
#     rec = recall_score(df_max_prob.loc[newly_annotated_files][col], pred.loc[newly_annotated_files][col])
#     f1 = f1_score(df_max_prob.loc[newly_annotated_files][col], pred.loc[newly_annotated_files][col])
#     print(f"{col} -- {prec:.2f} -- {rec:.2f} -- {f1:.2f}")

# # Initialize the DataFrame for storing F1 Scores
# f1_data = {}

# # Evaluating with the max probability 
# f1_data[("Max Prob", "Whole test set")] = [
#     f1_score(df_max_prob["Anth"], pred["Anth"]),
#     f1_score(df_max_prob["Bio"], pred["Bio"]),
#     f1_score(df_max_prob["Geo"], pred["Geo"]),
#     f1_score(df_max_prob["Sil"], pred["Sil"])
# ]

# f1_data[("Max Prob", "Subset")] = [
#     f1_score(df_max_prob.loc[newly_annotated_files]["Anth"], pred.loc[newly_annotated_files]["Anth"]),
#     f1_score(df_max_prob.loc[newly_annotated_files]["Bio"], predpred.loc[newly_annotated_files]["Bio"]),
#     f1_score(df_max_prob.loc[newly_annotated_files]["Geo"], pred.loc[newly_annotated_files]["Geo"]),
#     f1_score(df_max_prob.loc[newly_annotated_files]["Sil"], pred.loc[newly_annotated_files]["Sil"])
# ]



# Evaluating based on specialised and adpated thresholds
threshold = .4

# Load and prepare predictions for new thresholds
pred = pd.read_csv(predictions_path).set_index("filename")
pred = pred.groupby("filename").apply(
    lambda x: pd.Series({
        "Anth": (x["Anth"] > .5).sum() >= 5, # & ((x["Bio"] > .1).sum() <= 35) & ((x["Geo"] > .1).sum() <= 15), #((x["Anth"] > .1).sum() >= 25) & ((x["Bio"] > .1).sum() <= 35) & ((x["Geo"] > .1).sum() <= 15)
        "Bio": (x["Bio"] > .5).sum() >= 2, # 45
        "Geo": (x["Geo"] > .5).sum() >= 3, # 45
        "Sil": (x["Sil"] > threshold).sum() >= 55, # 55
    })
)
pred["Sil"] = pred.apply(lambda x: ~x["Anth"] & ~x["Bio"] & ~x["Geo"], axis=1)

# print(pred.head())
# print(pred["Sil"].value_counts())
# print("\n\n")

# Align indices
common_index = df.index.intersection(pred.index)
pred = pred.loc[common_index]
pred = pred.sort_index()


# Get samples where only 'Geo' is annotated
only_geo_annotated_mask = (
    (df["Geo"] == 1) &
    (df["Anth"] == 0) &
    (df["Bio"] == 0) &
    (df["Sil"] == 0)
)
only_geo_samples = df[only_geo_annotated_mask]

# From those, select samples where 'Geo' is predicted (regardless of other predictions)
pred_only_geo_subset = pred.loc[only_geo_samples.index]
geo_predicted_mask = pred_only_geo_subset["Geo"] == 1

# Final subset: samples where only Geo is annotated and Geo is predicted
final_index = pred_only_geo_subset[geo_predicted_mask].index
final_df = df.loc[final_index]
final_pred = pred.loc[final_index]

other_classes = ["Anth", "Bio", "Sil"]
print("\nOnly Geo")
for cls in other_classes:
    count = (final_pred[cls] == 1).sum()
    percentage = count / len(final_df) * 100 if len(final_df) > 0 else 0
    print(f"{cls} predicted in {count} of {len(final_df)} samples ({percentage:.2f}%)")


# Only Bio
only_bio_annotated_mask = (
    (df["Geo"] == 0) &
    (df["Anth"] == 0) &
    (df["Bio"] == 1) &
    (df["Sil"] == 0)
)
only_bio_samples = df[only_bio_annotated_mask]

# From those, select samples where 'Geo' is predicted (regardless of other predictions)
pred_only_bio_subset = pred.loc[only_bio_samples.index]
bio_predicted_mask = pred_only_bio_subset["Bio"] == 1

# Final subset: samples where only Geo is annotated and Geo is predicted
final_index = pred_only_bio_subset[bio_predicted_mask].index
final_df = df.loc[final_index]
final_pred = pred.loc[final_index]

print("\nOnly Bio")
other_classes = ["Anth", "Geo", "Sil"]
for cls in other_classes:
    count = (final_pred[cls] == 1).sum()
    percentage = count / len(final_df) * 100 if len(final_df) > 0 else 0
    print(f"{cls} predicted in {count} of {len(final_df)} samples ({percentage:.2f}%)")


# Falsely predicted biophony when no geophony or biophony were annotated
# no_bio_geo_annotated_mask = (
#     (df["Geo"] == 0) &
#     (df["Bio"] == 0)
# )
# no_bio_geo_samples = df[no_bio_geo_annotated_mask]

# # From those, select samples where 'Bio' is predicted
# pred_no_bio_geo_subset = pred.loc[no_bio_geo_samples.index]
# bio_predicted_mask = pred_no_bio_geo_subset["Bio"] == 1

# # Final subset: samples where only Geo is annotated and Geo is predicted
# final_index = pred_no_bio_geo_subset[bio_predicted_mask].index
# final_df = df.loc[final_index]
# final_pred = pred.loc[final_index]

# print("\nBio/Geo")
# other_classes = ["Anth", "Bio", "Geo", "Sil"]
# for cls in other_classes:
#     count = (final_pred[cls] == 1).sum()
#     percentage = count / len(final_df) * 100 if len(final_df) > 0 else 0
#     print(f"{cls} predicted in {count} of {len(final_df)} samples ({percentage:.2f}%)")



# Mask for samples where both Bio and Geo are not annotated
bio_geo_not_annotated_mask = (df["Bio"] == 0) & (df["Geo"] == 0)

# From these, find which have Bio predicted
bio_predicted_mask = pred["Bio"] == 1

# Combine the masks
bio_predicted_without_annotation_mask = bio_geo_not_annotated_mask & bio_predicted_mask

# Final count and rate
count = bio_predicted_without_annotation_mask.sum()
total = bio_geo_not_annotated_mask.sum()
percentage = count / total * 100 if total > 0 else 0

print(f"Bio predicted in {count} of {total} samples where neither Bio nor Geo are annotated ({percentage:.2f}%)")




# Mask for samples where Geo is annotated and predicted
geo_annotated_and_predicted_mask = (df["Geo"] == 1) & (pred["Geo"] == 1)

# From those, find where Bio is not annotated but predicted
bio_falsely_predicted_mask = (df["Bio"] == 0) & (pred["Bio"] == 1)

# Combine both conditions
false_bio_when_geo_true_mask = geo_annotated_and_predicted_mask & bio_falsely_predicted_mask

# Final count and rate
count = false_bio_when_geo_true_mask.sum()
total = geo_annotated_and_predicted_mask.sum()
percentage = count / total * 100 if total > 0 else 0

print(f"False Bio predicted in {count} of {total} samples where Geo was annotated and predicted ({percentage:.2f}%)")



# Mask for samples where Geo is annotated and predicted
anth_annotated_and_predicted_mask = (df["Anth"] == 1) & (pred["Anth"] == 1)

# From those, find where Bio is not annotated but predicted
bio_falsely_predicted_mask = (df["Bio"] == 0) & (pred["Bio"] == 1)

# Combine both conditions
false_bio_when_anth_true_mask = anth_annotated_and_predicted_mask & bio_falsely_predicted_mask

# Final count and rate
count = false_bio_when_anth_true_mask.sum()
total = anth_annotated_and_predicted_mask.sum()
percentage = count / total * 100 if total > 0 else 0

print(f"False Bio predicted in {count} of {total} samples where Anth was annotated and predicted ({percentage:.2f}%)")



