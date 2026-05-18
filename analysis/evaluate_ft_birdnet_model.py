import os
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from confidence_intervals_edansa import bootstrap_ci, bootstrap_macro_ci

version = "edansa"

if version == "besound":
    # BEsound
    df_test = pd.read_csv("./analysis/processed_annotations/old_annotations.csv")
    df_test = df_test.rename(columns={"filename": "file"})
    df_test = df_test[["file", "Anth", "Bio", "Geo"]]

    root = "/path/to/BEsound_custom_autotune_inference_results"
    remove_string = "/path/string/to/remove/"
elif version == "edansa":
    # Edansa
    df_test = pd.read_csv("/path/to/EDANSA-2019/test.csv")
    root = "/path/to/custom_autotune_inference_results"
    remove_string = "/path/string/to/remove/v2"

    df_test = df_test[["Clip Path", "Anth", "Bio", "Geo"]]
    df_test = df_test.rename(columns={"Clip Path": "file"})
else:
    print(f"The version '{version}' does not exist. Abort...")
    exit()

target_columns = ["Anth", "Bio", "Geo"]
dfs = []
# Get all the selection tables and bring them together in a dataframe
for dirpath, _, filenames in os.walk(root):
    for f in filenames:
        if f.endswith(".txt"):
            fp = os.path.join(dirpath, f)
            
            df = pd.read_csv(fp, sep="\t")
            
            # Optional: keep track of which file this came from
            # df["file"] = f
            df["file"] = df["Begin Path"].str.replace(remove_string, "")
            
            dfs.append(df)

# Create the dataframe and filter it based on a confidence threshold
df_all = pd.concat(dfs, ignore_index=True)
df_filtered = df_all[df_all["Confidence"] > 0.5]
# df_filtered = df_filtered[["file", "path", "Common Name", "Confidence"]]
df_filtered = df_filtered[["file", "Common Name", "Confidence"]]
df_unique = (
    df_filtered
    .sort_values("Confidence", ascending=False)
    .drop_duplicates(subset=["file", "Common Name"])
    .reset_index(drop=True)
)
print(df_unique)


df_wide = (
    df_unique
    .assign(value=1)  # mark presence
    .pivot_table(
        index="file", 
        columns="Common Name", 
        values="value", 
        fill_value=0
    )
    .reset_index()
)
print(df_wide)

# Align the dataframes and the order
df_wide = df_wide.set_index("file")
df_test = df_test.set_index("file")

common_files = df_test.index.intersection(df_wide.index)
df_test = df_test.loc[common_files].sort_index()
df_wide = df_wide.loc[common_files].sort_index()

# Get metrics
for col in target_columns:
    prec = precision_score(df_test[col], df_wide[col])
    rec = recall_score(df_test[col], df_wide[col])
    f1 = f1_score(df_test[col], df_wide[col])
    print(f"{col} -- {prec:.3f} -- {rec:.3f} -- {f1:.3f}")
print("Total F1-macro: ", f1_score(df_test[target_columns], df_wide[target_columns], average="macro"))
print(classification_report(df_test[target_columns], df_wide[target_columns]))


# Calculate confidence intervals
y_true = df_test[target_columns].values
y_pred_bin = df_wide[target_columns].values
ci = bootstrap_ci(y_true, y_pred_bin, labels=target_columns)
print("\n95% Confidence Intervals for per-class F1-scores:")
for label in ["Anth", "Bio", "Geo"]:
    print(f"{label}: {ci[label][0]:.2f} - {ci[label][1]:.2f}")

# Across classes
ci_low, ci_high = bootstrap_macro_ci(y_true, y_pred_bin, labels=target_columns)
print("Macro F1-score: ", f1_score(y_true, y_pred_bin, average="macro"))
print(f"95% CI for macro F1-score: 95% CI [{ci_low:.3f}, {ci_high:.3f}]")