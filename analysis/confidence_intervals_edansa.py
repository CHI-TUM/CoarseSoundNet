import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report, multilabel_confusion_matrix


def evaluate_metrics(y_true, y_pred, average="macro", labels=["Anth", "Bio", "Geo"]):
    f1s = []
    for col in labels:
        f1 = f1_score(y_true, y_pred, average=average)
        f1s.append(f1)
    return f1s


def bootstrap_ci(y_true, y_pred, n_bootstraps=1000, ci=95, average="macro", labels=["Anth", "Bio", "Geo"]):
    np.random.seed(42)
    n_samples = y_true.shape[0]
    stats = {label: [] for label in labels}

    for _ in range(n_bootstraps):
        sample_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_sample = y_true[sample_idx]
        y_pred_sample = y_pred[sample_idx]

        f1s = f1_score(y_true_sample, y_pred_sample, average=None)
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


def bootstrap_macro_ci(y_true, y_pred, n_bootstraps=1000, ci=95, labels=["Anth", "Bio", "Geo"]):
    np.random.seed(42)
    n_samples = y_true.shape[0]
    macro_f1_scores = []

    for _ in range(n_bootstraps):
        sample_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_sample = y_true[sample_idx]
        y_pred_sample = y_pred[sample_idx]

        # Compute macro F1 across all labels
        f1 = f1_score(y_true_sample, y_pred_sample, average='macro')
        macro_f1_scores.append(f1)

    # Compute CI bounds
    ci_low = (100 - ci) / 2
    ci_high = 100 - ci_low
    lower = np.percentile(macro_f1_scores, ci_low)
    upper = np.percentile(macro_f1_scores, ci_high)

    return lower, upper


if __name__=='__main__':
    path = "/path/to/model/training/folder"

    y_pred_probs = pd.read_csv(os.path.join(path, "_test/test_results.csv")).iloc[:, 2:]  # shape (6161, 30)
    y_pred_probs = y_pred_probs.drop(columns=["Sil"])
    y_true = np.load(os.path.join(path, "_test/test_targets.npy"))  # shape (6161, 30)
    y_true = y_true[:, :-1]
    
    target_names = y_pred_probs.columns.tolist()
    print("Target names: ", target_names)
    y_pred_probs = y_pred_probs.values
    print("Preds shape: ", y_pred_probs.shape)
    # Apply threshold to get predicted labels (binary)
    threshold = 0.5
    y_pred_bin = (y_pred_probs > threshold).astype(int)
    print(y_true.shape)


    # Generate classification report
    print(classification_report(y_true, y_pred_bin, target_names=target_names, zero_division=0))
    # Get classification report as a dict
    report_dict = classification_report(y_true, y_pred_bin, target_names=target_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    # report_df.to_csv(f"analysis/classification_report_{partition}.csv", float_format="%.2f")

    # Generate multilabel confusion matrix
    mlcm = multilabel_confusion_matrix(y_true, y_pred_bin)

    print("\nPer-class Confusion Matrices:")
    for i, matrix in enumerate(mlcm):
        tn, fp, fn, tp = matrix.ravel()
        print(f"\nSpecies: {target_names[i]}")
        print(f"  True Negatives : {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives : {tp}")


    # False positives: predicted 1 but actually 0
    false_positives = (y_pred_bin == 1) & (y_true == 0)

    # Initialize co-occurrence matrix
    co_confusion = np.zeros((len(target_names), len(target_names)), dtype=int)

    # Count co-occurrence of false positives and the actual class
    for i in range(len(y_true)):
        for pred_class in np.where(false_positives[i])[0]:
            for true_class in np.where(y_true[i])[0]:
                co_confusion[true_class, pred_class] += 1

    # Convert to DataFrame with labels
    co_df = pd.DataFrame(co_confusion, index=target_names, columns=target_names)
    # co_df.to_csv(f'analysis/co-confusion_{partition}.csv')

    print(co_df)

    # Plotting
    plt.figure(figsize=(14, 12))
    sns.heatmap(co_df, annot=False, fmt="d", cmap="viridis", linewidths=0.5)
    plt.title("Label Misclassification Co-occurrence Matrix\n(True Class → Predicted as)", fontsize=14)
    plt.xlabel("Predicted (False Positive) Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    # plt.savefig(f'analysis/co-confusion_Edansa-test.png')


    ci = bootstrap_ci(y_true, y_pred_bin, labels=target_names)
    print("\n95% Confidence Intervals for per-class F1-scores:")
    for label in ["Anth", "Bio", "Geo"]:
        print(f"{label}: {ci[label][0]:.2f} - {ci[label][1]:.2f}")
    
    # Across classes
    ci_low, ci_high = bootstrap_macro_ci(y_true, y_pred_bin, labels=target_names)
    print("Macro F1-score: ", f1_score(y_true, y_pred_bin, average="macro"))
    print(f"95% CI for macro F1-score: 95% CI [{ci_low:.3f}, {ci_high:.3f}]")

