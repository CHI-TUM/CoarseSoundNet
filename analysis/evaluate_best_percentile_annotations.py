import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve, average_precision_score


def bootstrap_ci(y_true, y_pred, n_bootstraps=1000, ci=95, labels=["Anth", "Bio", "Geo"], zero_division=0):
    np.random.seed(42)
    n_samples = y_true.shape[0]
    stats = {label: [] for label in labels}

    for _ in range(n_bootstraps):
        sample_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_sample = y_true.iloc[sample_idx]
        y_pred_sample = y_pred.iloc[sample_idx]

        f1s = f1_score(y_true_sample[labels], y_pred_sample[labels], average=None)
        for i, label in enumerate(labels):
            stats[label].append(f1s[i])

    ci_low = (100 - ci) / 2
    ci_high = 100 - ci_low
    ci_bounds = {}

    for label in labels:
        lower = np.percentile(stats[label], ci_low)
        upper = np.percentile(stats[label], ci_high)
        ci_bounds[label] = (lower, upper)

    return ci_bounds


def bootstrap_macro_ci(y_true, y_pred, n_bootstraps=1000, ci=95, labels=["Anth", "Bio", "Geo"], zero_division=0):
    np.random.seed(42)
    n_samples = y_true.shape[0]
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

def bootstrap_f1_cis(y_true, y_pred, labels, n_bootstraps=1000, ci=95, zero_division=0):
    """
    Bootstrap: returns per-class F1 CIs AND macro-F1 CI,
    computed from the exact same resampled rows each iteration.
    This guarantees the macro CI is consistent with the per-class CIs
    """
    np.random.seed(42)
    n_samples = len(y_true)
    per_class_scores = {label: [] for label in labels}
    macro_scores = []

    for _ in range(n_bootstraps):
        sample_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_sample = y_true.iloc[sample_idx]
        y_pred_sample = y_pred.iloc[sample_idx]

        f1s = [f1_score(y_true_sample[c], y_pred_sample[c], zero_division=zero_division) for c in labels]
        for label, f1 in zip(labels, f1s):
            per_class_scores[label].append(f1)
        macro_scores.append(np.mean(f1s))  # exactly consistent with per-class draws

    ci_low = (100 - ci) / 2
    ci_high = 100 - ci_low

    per_class_ci = {
        label: (np.percentile(scores, ci_low), np.percentile(scores, ci_high))
        for label, scores in per_class_scores.items()
    }
    macro_ci = (np.percentile(macro_scores, ci_low), np.percentile(macro_scores, ci_high))

    return per_class_ci, macro_ci


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


if __name__=='__main__':
    # Set CI computation True/False
    compute_CIs = True
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

    print("\n --- Adapted annotations with individual thresholds ---")

    # target_classes = ["Anth", "Bio", "Geo"]

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


    # Bootstrap confidence intervals for the learned-threshold results
    # if compute_CIs:
    #     ci = bootstrap_ci(df, pred_bin, labels=target_classes)
    #     print("\n95% CI for per-class F1 (learned thresholds):")
    #     for cls in target_classes:
    #         print(f"{cls}: {ci[cls][0]:.3f} - {ci[cls][1]:.3f}")

    #     macro_ci_low, macro_ci_high = bootstrap_macro_ci(df, pred_bin, labels=target_classes)
    #     print(f"95% CI for macro F1 (learned thresholds): {macro_ci_low:.3f} - {macro_ci_high:.3f}")

    if compute_CIs:
        per_class_ci, (macro_ci_low, macro_ci_high) = bootstrap_f1_cis(df, pred_bin, labels=target_classes)

        print("\n95% CI for per-class F1 (learned thresholds):")
        for cls in target_classes:
            print(f"{cls}: {per_class_ci[cls][0]:.3f} - {per_class_ci[cls][1]:.3f}")

        print(f"95% CI for macro F1 (learned thresholds): {macro_ci_low:.3f} - {macro_ci_high:.3f}")


    print("\n############################################################")
    print("--- Filename groups for error analysis ---\n")

    # Group 1: Anthropophony masked by biophony or geophony
    # (Anth annotated but missed by model, while Bio and/or Geo are actually present)
    mask_g1 = (df["Anth"] == 1) & (pred_bin["Anth"] == 0) & ((df["Bio"] == 1) | (df["Geo"] == 1))
    g1_files = df.index[mask_g1].tolist()
    print(f"Group 1 - Anth masked by Bio/Geo ({len(g1_files)} files):")
    # for f in g1_files:
    #     co = []
    #     if df.loc[f, "Bio"] == 1:
    #         co.append("Bio")
    #     if df.loc[f, "Geo"] == 1:
    #         co.append("Geo")
    #     print(f"  {f}  (co-occurring: {', '.join(co)})")

    # Group 2: Geophony confused with silence
    # (Geo annotated but missed, AND model predicted nothing else either -> predicted Sil)
    mask_g2 = (df["Geo"] == 1) & (pred_bin["Geo"] == 0) & (pred_bin["Sil"] == 1)
    g2_files = df.index[mask_g2].tolist()
    print(f"\nGroup 2 - Geo confused with silence ({len(g2_files)} files):")
    # for f in g2_files:
    #     print(f"  {f}")

    # Group 3: Biophony not recognized even though annotated
    mask_g3 = (df["Bio"] == 1) & (pred_bin["Bio"] == 0)
    g3_files = df.index[mask_g3].tolist()
    print(f"\nGroup 3 - Bio missed ({len(g3_files)} files):")
    # for f in g3_files:
    #     print(f"  {f}")

    # Optional: dump each group to its own CSV for easier inspection
    pd.Series(g1_files, name="filename").to_csv("./analysis/error_analysis_filenames/group1_anth_masked.csv", index=False)
    pd.Series(g2_files, name="filename").to_csv("./analysis/error_analysis_filenames/group2_geo_confused_with_silence.csv", index=False)
    pd.Series(g3_files, name="filename").to_csv("./analysis/error_analysis_filenames/group3_bio_missed.csv", index=False)
    print("\n############################################################")
        

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
    predictions_path = f"/path/to/BESound_predictions_dataset_variations_smallSteps/balanced/{version}/{dataset}/results.csv"

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
    predictions_path = f"/path/to/BESound_predictions_dataset_variations_smallSteps/balanced/{version}/{dataset}/results.csv"

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


    if compute_CIs:
        per_class_ci, (macro_ci_low, macro_ci_high) = bootstrap_f1_cis(df, pred, labels=target_classes)

        print("\n95% CI for per-class F1 (learned thresholds):")
        for cls in target_classes:
            print(f"{cls}: {per_class_ci[cls][0]:.3f} - {per_class_ci[cls][1]:.3f}")

        print(f"95% CI for macro F1 (learned thresholds): {macro_ci_low:.3f} - {macro_ci_high:.3f}")