import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

# bootstrap_f1_cis and best_threshold_by_f1 are safe to import: that module's script
# logic is wrapped in `if __name__ == "__main__":`, so importing it only pulls in the
# function definitions and does not re-run its own analysis.
from evaluate_best_percentile_annotations import bootstrap_f1_cis, best_threshold_by_f1

# target_classes = ["Anth", "Bio", "Geo", "Sil"]
target_classes = ["Anth", "Bio", "Geo"]
compute_CIs = True

# --- Cross-validation settings ---
n_folds = 3
cv_random_state = 42  # fixed seed so the group shuffle is reproducible


def extract_plot_id(filename):
    """
    Filenames look like '<startnumber>-<date>_<plot>.wav', e.g.
    '0000-01042016_223.wav' -> plot id '223'. Used as the CV group so all
    recordings from the same plot land in the same fold (never split across
    train/test).
    """
    stem = os.path.splitext(filename)[0]
    return stem.split("_")[-1]


############################################################
# This script combines TWO choices that were previously each learned on the
# full dataset and then evaluated on that same full dataset:
#   1) PDA - which annotation percentage-variant (baseline/p5/p10/p25) to use
#      per class
#   2) CST - the class-specific threshold used to binarize predictions per class
#
# Both are now learned per fold using ONLY the training folds, and applied to
# the held-out fold. This means we start from the four separate annotation
# variant files (not a single pre-merged CSV), since the PDA choice itself has
# to be re-learned per fold, just like the CST threshold.
############################################################

print("\n--- Loading annotation variants and predictions ---")

df_base = pd.read_csv("./analysis/processed_annotations_percentages/baseline_annotations.csv").set_index("filename")
df_p5 = pd.read_csv("./analysis/processed_annotations_percentages/adapted_annotations_p5.csv").set_index("filename")
df_p10 = pd.read_csv("./analysis/processed_annotations_percentages/adapted_annotations_p10.csv").set_index("filename")
df_p25 = pd.read_csv("./analysis/processed_annotations_percentages/adapted_annotations_p25.csv").set_index("filename")

annotations = {
    "baseline": df_base,
    "p5": df_p5,
    "p10": df_p10,
    "p25": df_p25,
}

aggregation_type = "max"
version = "withSilence"
dataset = "pam-data"
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
    pred = pred.groupby("filename").max() if aggregation_type == "max" else pred.groupby("filename").mean()

# Keep raw probabilities for the final export, before anything gets aligned/binarized
max_probs_df = pred.copy()

# Align every annotation variant and the predictions to a common index
for variant, df_ann in annotations.items():
    common_index = pred.index.intersection(df_ann.index)
    annotations[variant] = df_ann.loc[common_index].sort_index()
pred = pred.loc[common_index].sort_index()
max_probs_df = max_probs_df.loc[common_index].sort_index()
max_probs_df = max_probs_df.rename(columns={col: f"{col}_max_prob" for col in max_probs_df.columns})

print("pred shape:", pred.shape)
for variant, df_ann in annotations.items():
    print(f"{variant} shape:", df_ann.shape)


############################################################
# Grouped 3-fold CV: for each fold, select the best annotation variant (PDA)
# per class AND learn its CST threshold, using train data only, then apply both
# to the held-out test fold.
############################################################

filenames = pred.index.to_numpy()
groups = np.array([extract_plot_id(f) for f in filenames])
print(f"\n{len(filenames)} recordings across {len(np.unique(groups))} unique plots")

gkf = GroupKFold(n_splits=n_folds)
# shuffle group order first since GroupKFold assigns groups to folds in the order
# it encounters them (not randomly) unless we randomize the input order ourselves
rng = np.random.RandomState(cv_random_state)
shuffle_pos = rng.permutation(len(filenames))
filenames = filenames[shuffle_pos]
groups = groups[shuffle_pos]

fold_results = []
fold_selections = []   # per fold: {class: selected_variant}
fold_thresholds = []   # per fold: {class: learned_threshold}

# Out-of-fold containers: every sample gets exactly one ground-truth row (from
# the PDA variant selected without seeing it) and one prediction row (from the
# CST threshold learned without seeing it)
oof_truth = pd.DataFrame(index=pred.index, columns=target_classes, dtype=float)
oof_pred = pd.DataFrame(index=pred.index, columns=target_classes, dtype=float)

for fold_idx, (train_pos, test_pos) in enumerate(gkf.split(filenames, groups=groups), start=1):
    train_files = filenames[train_pos]
    test_files = filenames[test_pos]

    train_plots = set(groups[train_pos])
    test_plots = set(groups[test_pos])
    assert not (train_plots & test_plots), "Plot overlap between train/test fold - grouping failed!"
    print(f"\nFold {fold_idx}/{n_folds} -- train n={len(train_files)} ({len(train_plots)} plots), "
          f"test n={len(test_files)} ({len(test_plots)} plots)")

    pred_train = pred.loc[train_files]
    pred_test = pred.loc[test_files]

    # --- Step 1 (PDA): select best annotation variant per class on train (fixed 0.5 threshold) ---
    pred_train_05 = pred_train[target_classes].apply(lambda col: (col > 0.5)).astype(int)
    selection_fold = {}
    for cls in target_classes:
        best_variant, best_f1 = None, -1.0
        for variant, df_ann in annotations.items():
            f1 = f1_score(df_ann.loc[train_files, cls], pred_train_05[cls], zero_division=0)
            if f1 > best_f1:
                best_variant, best_f1 = variant, f1
        selection_fold[cls] = best_variant
    fold_selections.append(selection_fold)
    print("  Selected variant per class (PDA):", selection_fold)

    # --- Step 2 (CST): learn best-F1 threshold per class on train, against the selected variant ---
    thresholds_fold = {}
    for cls in target_classes:
        variant = selection_fold[cls]
        y_true_train = annotations[variant].loc[train_files, cls].values
        y_score_train = pred_train[cls].values
        thresholds_fold[cls] = best_threshold_by_f1(y_true_train, y_score_train)
    fold_thresholds.append(thresholds_fold)
    print("  Learned CST thresholds:", {k: round(v, 4) for k, v in thresholds_fold.items()})

    # --- Apply both selections to the held-out test fold ---
    df_test_selected = pd.DataFrame(index=test_files, columns=target_classes, dtype=float)
    pred_test_bin = pd.DataFrame(index=test_files, columns=target_classes, dtype=float)
    for cls in target_classes:
        variant = selection_fold[cls]
        df_test_selected[cls] = annotations[variant].loc[test_files, cls]
        pred_test_bin[cls] = (pred_test[cls] > thresholds_fold[cls]).astype(int)

    oof_truth.loc[test_files, target_classes] = df_test_selected[target_classes]
    oof_pred.loc[test_files, target_classes] = pred_test_bin[target_classes]

    print("  Class -- Precision -- Recall -- F1-Score")
    fold_metrics = {}
    for cls in target_classes:
        prec = precision_score(df_test_selected[cls], pred_test_bin[cls], zero_division=0)
        rec = recall_score(df_test_selected[cls], pred_test_bin[cls], zero_division=0)
        f1 = f1_score(df_test_selected[cls], pred_test_bin[cls], zero_division=0)
        fold_metrics[cls] = {"precision": prec, "recall": rec, "f1": f1}
        print(f"  {cls} -- {prec:.3f} -- {rec:.3f} -- {f1:.3f}")

    macro_f1 = f1_score(df_test_selected[target_classes], pred_test_bin[target_classes], average="macro", zero_division=0)
    fold_metrics["macro_f1"] = macro_f1
    print(f"  Fold Total F1-macro: {macro_f1:.3f}")

    fold_results.append(fold_metrics)


# --- Aggregate across folds (mean +/- std) ---
print(f"\n--- Aggregated CV results (mean +/- std over {n_folds} folds) ---")
print("Class -- Precision -- Recall -- F1-Score")
for cls in target_classes:
    precs = [fr[cls]["precision"] for fr in fold_results]
    recs = [fr[cls]["recall"] for fr in fold_results]
    f1s = [fr[cls]["f1"] for fr in fold_results]
    print(f"{cls} -- {np.mean(precs):.3f} +/- {np.std(precs):.3f}"
          f" -- {np.mean(recs):.3f} +/- {np.std(recs):.3f}"
          f" -- {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")

macro_f1s = [fr["macro_f1"] for fr in fold_results]
print(f"Total F1-macro: {np.mean(macro_f1s):.3f} +/- {np.std(macro_f1s):.3f}")

print("\nPer-fold selected annotation variant (PDA):")
for i, sel in enumerate(fold_selections, start=1):
    print(f"  Fold {i}: {sel}")

print("\nPer-fold learned CST thresholds:")
for i, thr in enumerate(fold_thresholds, start=1):
    rounded = {k: round(v, 4) for k, v in thr.items()}
    print(f"  Fold {i}: {rounded}")


############################################################
# Pooled out-of-fold evaluation: every sample was scored using a PDA variant and
# a CST threshold learned only from the other two folds, so pooling gives one
# combined-approach evaluation over the full dataset that is still fully
# held-out per-sample. This is the number to report as "PDA + CST, CV".
############################################################
print("\n--- Pooled out-of-fold evaluation (combined PDA + CST) ---")

oof_truth = oof_truth.astype(int)
oof_pred = oof_pred.astype(int)
# Derive Sil as the complement of the other three, same convention as elsewhere
oof_pred["Sil"] = (~(oof_pred["Anth"].astype(bool) | oof_pred["Bio"].astype(bool) | oof_pred["Geo"].astype(bool))).astype(int)

print("Class -- Precision -- Recall -- F1-Score")
for cls in target_classes:
    prec = precision_score(oof_truth[cls], oof_pred[cls], zero_division=0)
    rec = recall_score(oof_truth[cls], oof_pred[cls], zero_division=0)
    f1 = f1_score(oof_truth[cls], oof_pred[cls], zero_division=0)
    print(f"{cls} -- {prec:.3f} -- {rec:.3f} -- {f1:.3f}")

macro_f1 = f1_score(oof_truth[target_classes], oof_pred[target_classes], average="macro", zero_division=0)
print("Total F1-macro:", round(macro_f1, 3))

if compute_CIs:
    per_class_ci, (macro_ci_low, macro_ci_high) = bootstrap_f1_cis(oof_truth, oof_pred, labels=target_classes)

    print("\n95% CI for per-class F1 (pooled out-of-fold, PDA + CST):")
    for cls in target_classes:
        print(f"{cls}: {per_class_ci[cls][0]:.3f} - {per_class_ci[cls][1]:.3f}")

    print(f"95% CI for macro F1 (pooled out-of-fold, PDA + CST): {macro_ci_low:.3f} - {macro_ci_high:.3f}")


# ############################################################
# # Error analysis and final combined CSV export, now run on the pooled
# # out-of-fold ground truth / predictions (oof_truth / oof_pred) instead of a
# # single full-data fit, so every row reflects a held-out decision.
# ############################################################
# print("\n############################################################")
# print("--- Filename groups for error analysis (out-of-fold) ---\n")

# df = oof_truth  # keep the variable name used by the downstream logic below
# pred_bin = oof_pred

# # Group 1: Anthropophony masked by biophony or geophony
# mask_g1 = (df["Anth"] == 1) & (pred_bin["Anth"] == 0) & ((df["Bio"] == 1) | (df["Geo"] == 1))
# g1_files = df.index[mask_g1].tolist()
# print(f"Group 1 - Anth masked by Bio/Geo ({len(g1_files)} files)")

# # Group 2: Geophony confused with silence
# mask_g2 = (df["Geo"] == 1) & (pred_bin["Geo"] == 0) & (pred_bin["Sil"] == 1)
# g2_files = df.index[mask_g2].tolist()
# print(f"Group 2 - Geo confused with silence ({len(g2_files)} files)")

# # Group 3: Biophony not recognized even though annotated
# mask_g3 = (df["Bio"] == 1) & (pred_bin["Bio"] == 0)
# g3_files = df.index[mask_g3].tolist()
# print(f"Group 3 - Bio missed ({len(g3_files)} files)")

# os.makedirs("./analysis/error_analysis_filenames", exist_ok=True)
# # pd.Series(g1_files, name="filename").to_csv("./analysis/error_analysis_filenames/group1_anth_masked_cv.csv", index=False)
# # pd.Series(g2_files, name="filename").to_csv("./analysis/error_analysis_filenames/group2_geo_confused_with_silence_cv.csv", index=False)
# # pd.Series(g3_files, name="filename").to_csv("./analysis/error_analysis_filenames/group3_bio_missed_cv.csv", index=False)
# print("\n############################################################")

# # Combine into a single summary dataframe, mirroring the original script's
# # output but built entirely from out-of-fold results
# df_export = df.rename(columns={col: f"gt_new_{col}" for col in df.columns})
# pred_export = pred_bin.rename(columns={col: f"pred_new_{col}" for col in pred_bin.columns})
# df_combined = pd.concat([max_probs_df, df_export, pred_export], axis=1)

# summary_df = pd.read_csv("/nas/staff/data_work/AG/HearTheSpecies/BESound_indices_summary.csv").set_index("filename")
# summary_df = summary_df.iloc[:, :10]

# common_index = summary_df.index.intersection(df_combined.index)
# df_combined = df_combined.loc[common_index].sort_index()
# summary_df = summary_df.loc[common_index].sort_index()

# summary_df = pd.concat([summary_df, df_combined], axis=1).reset_index()
# summary_df.to_csv("/nas/staff/data_work/AG/HearTheSpecies/BESound_indices_summary_paperVersion_cv.csv", index=False)
# print("\nSaved out-of-fold summary CSV to BESound_indices_summary_paperVersion_cv.csv")