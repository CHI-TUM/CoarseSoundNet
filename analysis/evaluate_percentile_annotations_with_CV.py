import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold
from evaluate_best_percentile_annotations import bootstrap_f1_cis

# target_classes = ["Anth", "Bio", "Geo", "Sil"]
target_classes = ["Anth", "Bio", "Geo"]
compute_CIs = True

# --- Cross-validation settings ---
n_folds = 3
cv_random_state = 42  # fixed seed so the fold split is reproducible

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

if compute_CIs:
    per_class_ci, (macro_ci_low, macro_ci_high) = bootstrap_f1_cis(df_best_p, pred_base, labels=target_classes)

    print("\n95% CI for per-class F1 (learned thresholds):")
    for cls in target_classes:
        print(f"{cls}: {per_class_ci[cls][0]:.3f} - {per_class_ci[cls][1]:.3f}")

    print(f"95% CI for macro F1 (learned thresholds): {macro_ci_low:.3f} - {macro_ci_high:.3f}")


################
# Cross-validated selection of the best annotation percentage per class.
# The block above picks the "best percentage" variant per class (baseline/p5/p10/p25)
# by eyeballing F1 computed on the FULL dataset, then evaluates that same choice on
# the full dataset again - i.e. the selection and the evaluation use the same data.
# Here we do it properly with 3-fold CV: on each fold, the best variant per class is
# selected using only the training folds (fixed 0.5 threshold predictions), then that
# selection is evaluated on the held-out fold.
print("\n--- Cross-validated selection of best annotation percentage (0.5 threshold) ---")

filenames_pct = pred_base.index.to_numpy()
kf_pct = KFold(n_splits=n_folds, shuffle=True, random_state=cv_random_state)

pct_fold_results = []
pct_fold_selections = []
# out-of-fold "ground truth": for each held-out sample/class, use the annotation
# variant that was selected using only the other two folds
oof_truth = pd.DataFrame(index=pred_base.index, columns=target_classes, dtype=float)

for fold_idx, (train_pos, test_pos) in enumerate(kf_pct.split(filenames_pct), start=1):
    train_files = filenames_pct[train_pos]
    test_files = filenames_pct[test_pos]

    pred_train = pred_base.loc[train_files]
    pred_test = pred_base.loc[test_files]

    print(f"\nFold {fold_idx}/{n_folds} -- train n={len(train_files)}, test n={len(test_files)}")

    # Select, per class, the annotation variant with the highest F1 on the training fold
    selection_fold = {}
    for cls in target_classes:
        best_variant, best_f1 = None, -1.0
        for variant, df_ann in annotations.items():
            f1 = f1_score(df_ann.loc[train_files, cls], pred_train[cls], zero_division=0)
            if f1 > best_f1:
                best_variant, best_f1 = variant, f1
        selection_fold[cls] = best_variant
    pct_fold_selections.append(selection_fold)
    print("  Selected variant per class:", selection_fold)

    # Evaluate the selection on the held-out test fold
    print("  Class -- Precision -- Recall -- F1-Score")
    fold_metrics = {}
    df_test_selected = pd.DataFrame(index=test_files, columns=target_classes, dtype=float)
    for cls in target_classes:
        variant = selection_fold[cls]
        y_true_test = annotations[variant].loc[test_files, cls]
        df_test_selected[cls] = y_true_test

        prec = precision_score(y_true_test, pred_test[cls], zero_division=0)
        rec = recall_score(y_true_test, pred_test[cls], zero_division=0)
        f1 = f1_score(y_true_test, pred_test[cls], zero_division=0)
        fold_metrics[cls] = {"precision": prec, "recall": rec, "f1": f1}
        print(f"  {cls} -- {prec:.3f} -- {rec:.3f} -- {f1:.3f}")

    macro_f1 = f1_score(df_test_selected[target_classes], pred_test[target_classes], average="macro", zero_division=0)
    fold_metrics["macro_f1"] = macro_f1
    print(f"  Fold Total F1-macro: {macro_f1:.3f}")

    oof_truth.loc[test_files, target_classes] = df_test_selected[target_classes]
    pct_fold_results.append(fold_metrics)

# --- Aggregate across folds (mean +/- std over the n_folds) ---
print(f"\n--- Aggregated CV results for annotation-percentage selection (mean +/- std over {n_folds} folds) ---")
print("Class -- Precision -- Recall -- F1-Score")
for cls in target_classes:
    precs = [fr[cls]["precision"] for fr in pct_fold_results]
    recs = [fr[cls]["recall"] for fr in pct_fold_results]
    f1s = [fr[cls]["f1"] for fr in pct_fold_results]
    print(f"{cls} -- {np.mean(precs):.3f} +/- {np.std(precs):.3f}"
          f" -- {np.mean(recs):.3f} +/- {np.std(recs):.3f}"
          f" -- {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")

macro_f1s = [fr["macro_f1"] for fr in pct_fold_results]
print(f"Total F1-macro: {np.mean(macro_f1s):.3f} +/- {np.std(macro_f1s):.3f}")

# --- Pooled out-of-fold evaluation ---
# Every sample's "ground truth" comes from a variant selection that never saw that
# sample, so pooling gives one evaluation over the full dataset that is still fully
# held-out per-sample.
print("\n--- Pooled out-of-fold evaluation (each sample's variant chosen without seeing it) ---")
print("Class -- Precision -- Recall -- F1-Score")
for cls in target_classes:
    prec = precision_score(oof_truth[cls], pred_base[cls], zero_division=0)
    rec = recall_score(oof_truth[cls], pred_base[cls], zero_division=0)
    f1 = f1_score(oof_truth[cls], pred_base[cls], zero_division=0)
    print(f"{cls} -- {prec:.3f} -- {rec:.3f} -- {f1:.3f}")

macro_f1 = f1_score(oof_truth[target_classes], pred_base[target_classes], average="macro", zero_division=0)
print("Total F1-macro:", round(macro_f1, 3))

if compute_CIs:
    per_class_ci, (macro_ci_low, macro_ci_high) = bootstrap_f1_cis(oof_truth, pred_base, labels=target_classes)

    print("\n95% CI for per-class F1 (pooled out-of-fold, percentage selection):")
    for cls in target_classes:
        print(f"{cls}: {per_class_ci[cls][0]:.3f} - {per_class_ci[cls][1]:.3f}")

    print(f"95% CI for macro F1 (pooled out-of-fold, percentage selection): {macro_ci_low:.3f} - {macro_ci_high:.3f}")

print("\nPer-fold selected annotation variant:")
for i, sel in enumerate(pct_fold_selections, start=1):
    print(f"  Fold {i}: {sel}")


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
# Calculate results for CST (class-specific thresholds) using 3-fold cross-validation
# on the baseline annotations. Thresholds are learned (via Youden's J on the ROC
# curve) on 2 folds and evaluated on the held-out 3rd fold. Each fold serves as the
# test fold exactly once.
print("\n--- Calculate with CST via 3-fold cross-validation on the baseline annotations ---")

pred_cv = pd.read_csv(predictions_path).set_index("filename")
pred_cv = pred_cv.drop(columns=["prediction", "output"])
if "offset" in pred_cv.columns:
    pred_cv = pred_cv[~pred_cv["offset"].str.contains("majority", case=False, na=False)]
    pred_cv.drop(columns=["offset"], inplace=True)
    if aggregation_type == "max":
        pred_cv = pred_cv.groupby("filename").max()
    else:
        pred_cv = pred_cv.groupby("filename").mean()
if "Sil" in pred_cv.columns:
    pred_cv = pred_cv.drop(columns=["Sil"])

# Align indices between predictions and baseline annotations
common_index_cv = pred_cv.index.intersection(df_base.index)
df_base_cv = df_base.loc[common_index_cv].sort_index()
pred_cv = pred_cv.loc[common_index_cv].sort_index()

filenames = df_base_cv.index.to_numpy()
kf = KFold(n_splits=n_folds, shuffle=True, random_state=cv_random_state)

fold_results = []
fold_thresholds = []
# out-of-fold predictions: each sample gets predicted exactly once, using thresholds
# learned only from the other two folds
oof_pred = pd.DataFrame(index=df_base_cv.index, columns=target_classes, dtype=float)

for fold_idx, (train_pos, test_pos) in enumerate(kf.split(filenames), start=1):
    train_files = filenames[train_pos]
    test_files = filenames[test_pos]

    df_train = df_base_cv.loc[train_files]
    df_test = df_base_cv.loc[test_files]
    pred_train = pred_cv.loc[train_files]
    pred_test = pred_cv.loc[test_files]

    print(f"\nFold {fold_idx}/{n_folds} -- train n={len(train_files)}, test n={len(test_files)}")

    # Learn class-specific thresholds on the training folds only (Youden's J)
    thresholds_fold = {}
    for cls in target_classes:
        fpr, tpr, thr = roc_curve(df_train[cls], pred_train[cls])
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        thresholds_fold[cls] = thr[best_idx]
    fold_thresholds.append(thresholds_fold)
    print("  Learned thresholds:", {k: round(v, 3) for k, v in thresholds_fold.items()})

    # Apply thresholds to the held-out test fold
    pred_test_bin = pred_test.copy()
    pred_test_bin = pred_test_bin.apply(lambda col: (col > thresholds_fold[col.name]))
    pred_test_bin["Sil"] = ~(pred_test_bin["Anth"] | pred_test_bin["Bio"] | pred_test_bin["Geo"])
    pred_test_bin = pred_test_bin.astype(int)

    oof_pred.loc[test_files, target_classes] = pred_test_bin[target_classes]

    print("  Class -- Precision -- Recall -- F1-Score")
    fold_metrics = {}
    for cls in target_classes:
        prec = precision_score(df_test[cls], pred_test_bin[cls], zero_division=0)
        rec = recall_score(df_test[cls], pred_test_bin[cls], zero_division=0)
        f1 = f1_score(df_test[cls], pred_test_bin[cls], zero_division=0)
        fold_metrics[cls] = {"precision": prec, "recall": rec, "f1": f1}
        print(f"  {cls} -- {prec:.3f} -- {rec:.3f} -- {f1:.3f}")

    macro_f1 = f1_score(df_test[target_classes], pred_test_bin[target_classes], average="macro", zero_division=0)
    fold_metrics["macro_f1"] = macro_f1
    print(f"  Fold Total F1-macro: {macro_f1:.3f}")

    fold_results.append(fold_metrics)

# --- Aggregate across folds (mean +/- std over the n_folds) ---
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

# --- Pooled out-of-fold evaluation ---
# Every sample was predicted exactly once, with a threshold learned from data it was
# not part of. Pooling these gives a single CST evaluation over the full dataset
# that is still fully held-out per-sample.
print("\n--- Pooled out-of-fold CST evaluation (each sample predicted exactly once) ---")
oof_pred = oof_pred.astype(int)
print("Class -- Precision -- Recall -- F1-Score")
for cls in target_classes:
    prec = precision_score(df_base_cv[cls], oof_pred[cls], zero_division=0)
    rec = recall_score(df_base_cv[cls], oof_pred[cls], zero_division=0)
    f1 = f1_score(df_base_cv[cls], oof_pred[cls], zero_division=0)
    print(f"{cls} -- {prec:.3f} -- {rec:.3f} -- {f1:.3f}")

macro_f1 = f1_score(df_base_cv[target_classes], oof_pred[target_classes], average="macro", zero_division=0)
print("Total F1-macro:", round(macro_f1, 3))

if compute_CIs:
    per_class_ci, (macro_ci_low, macro_ci_high) = bootstrap_f1_cis(df_base_cv, oof_pred, labels=target_classes)

    print("\n95% CI for per-class F1 (pooled out-of-fold, CST):")
    for cls in target_classes:
        print(f"{cls}: {per_class_ci[cls][0]:.3f} - {per_class_ci[cls][1]:.3f}")

    print(f"95% CI for macro F1 (pooled out-of-fold, CST): {macro_ci_low:.3f} - {macro_ci_high:.3f}")

print("\nPer-fold learned thresholds:")
for i, thr in enumerate(fold_thresholds, start=1):
    rounded = {k: round(v, 3) for k, v in thr.items()}
    print(f"  Fold {i}: {rounded}")