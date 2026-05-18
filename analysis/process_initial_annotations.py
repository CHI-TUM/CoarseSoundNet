"""
Prepare the new annotations from Sandra and Dominik for evaluation.
Be aware that they only updated the rows where model and annotations were wrong.
The entries where model and annoations were agreeing were not reevaluated.
"""

import pandas as pd
import numpy as np


# load and process former annotations (by HiWis)
annotations_path = "./annotations_BEForest.csv"
df = pd.read_csv(annotations_path)
df.rename(columns={"file": "filename"}, inplace=True)
df["filename"] += ".wav"
df = df.set_index("filename")
df["duration"] = df["End Time (s)"] - df["Begin Time (s)"]
print()
print(df.head())
print(df.shape)
print()
print(df.columns)
print(df["Superclass"].unique())


def map_classes_baseline(x):
    anth = int(np.sum(x["Superclass"] == "Anthropophony") > 0)
    bio = int(np.sum(x["Superclass"] == "Biophony") > 0)
    geo = int(np.sum(x["Superclass"] == "Geophony") > 0)
    sil = int((anth + bio + geo == 0))
    return pd.Series(
        {
            "Anth": anth,
            "Bio": bio,
            "Geo": geo,
            "Sil": sil,
        }
    )


df_baseline = df.groupby("filename").apply(map_classes_baseline)
print(df_baseline.head())
print(df_baseline.shape)


# Compute percentiles based on all annotation durations
percentiles = [5, 10, 25]
durations_by_class = {
    "Anth": df[df["Superclass"] == "Anthropophony"]["duration"],
    "Bio": df[df["Superclass"] == "Biophony"]["duration"],
    "Geo": df[df["Superclass"] == "Geophony"]["duration"],
    "Sil": df[df["Superclass"] == "Silence"]["duration"],
}

# Consider all annotation durations
thresholds = {
    # p: {cls: np.percentile(dur, p) for cls, dur in durations_by_class.items()} for p in percentiles
    p: {cls: 60 * p / 100 for cls, dur in durations_by_class.items()}
    for p in percentiles
}
print("\nThresholds all annotations: ", thresholds)
# print(df.loc[df["Superclass"] == "Anthropophony", "duration"].describe())
# exit()


def map_classes_percentile(x, threshold):
    # anth = int((x.loc[x["Superclass"] == "Anthropophony", "duration"] >= threshold["Anth"]).any())
    # bio = int((x.loc[x["Superclass"] == "Biophony", "duration"] >= threshold["Bio"]).any())
    # geo = int((x.loc[x["Superclass"] == "Geophony", "duration"] >= threshold["Geo"]).any())
    anth = int(
        (
            x.loc[x["Superclass"] == "Anthropophony", "duration"].sum()
            >= threshold["Anth"]
        )
    )
    bio = int(
        (x.loc[x["Superclass"] == "Biophony", "duration"].sum() >= threshold["Bio"])
    )
    geo = int(
        (x.loc[x["Superclass"] == "Geophony", "duration"].sum() >= threshold["Geo"])
    )
    sil = int((anth + bio + geo == 0))
    return pd.Series(
        {
            "Anth": anth,
            "Bio": bio,
            "Geo": geo,
            "Sil": sil,
        }
    )


df_p5 = df.groupby("filename").apply(map_classes_percentile, threshold=thresholds[5])
df_p10 = df.groupby("filename").apply(map_classes_percentile, threshold=thresholds[10])
df_p25 = df.groupby("filename").apply(map_classes_percentile, threshold=thresholds[25])

print("\n5%:")
print(df_p5.head())
print("\n10%:")
print(df_p10.head())
print("\n25%:")
print(df_p25.head())

# df_baseline.reset_index().to_csv("./analysis/processed_annotations_percentiles/baseline_annotations.csv", index=False)
# df_p5.reset_index().to_csv("./analysis/processed_annotations_percentiles/adapted_annotations_p5.csv", index=False)
# df_p10.reset_index().to_csv("./analysis/processed_annotations_percentiles/adapted_annotations_p10.csv", index=False)
# df_p25.reset_index().to_csv("./analysis/processed_annotations_percentiles/adapted_annotations_p25.csv", index=False)

df_baseline.reset_index().to_csv(
    "./analysis/processed_annotations_percentages/baseline_annotations.csv", index=False
)
df_p5.reset_index().to_csv(
    "./analysis/processed_annotations_percentages/adapted_annotations_p5.csv",
    index=False,
)
df_p10.reset_index().to_csv(
    "./analysis/processed_annotations_percentages/adapted_annotations_p10.csv",
    index=False,
)
df_p25.reset_index().to_csv(
    "./analysis/processed_annotations_percentages/adapted_annotations_p25.csv",
    index=False,
)


# DO IT FOR THE BEST THRESHOLDS based on evaluate_percentile_annotations.py
best_thresholds = {
    "Anth": 15, # 6,  # PR: p5 or p10; ROC: p25 => p10
    # "Bio": 3,  # PR + ROC: baseline, p5, or p10 => p5
    "Bio": 0,  # PR + ROC: baseline, p5, or p10 => p5
    "Geo": 15, # 6,  # PR: p10 ROC: p25, then p10 => p10
}


def map_classes_best_percentage(x, threshold):
    anth = int(
        (
            x.loc[x["Superclass"] == "Anthropophony", "duration"].sum()
            >= best_thresholds["Anth"]
        )
    )
    bio = int(
        (
            x.loc[x["Superclass"] == "Biophony", "duration"].sum()
            > 0 # Based on the evaluations it didn't make a noticable difference whether we applied a threshold or not. Also Sandra said it would be better to have no constraint for Bio.
            # >= best_thresholds["Bio"]
        )
    )
    geo = int(
        (
            x.loc[x["Superclass"] == "Geophony", "duration"].sum()
            >= best_thresholds["Geo"]
        )
    )
    sil = int((anth + bio + geo == 0))
    return pd.Series(
        {
            "Anth": anth,
            "Bio": bio,
            "Geo": geo,
            "Sil": sil,
        }
    )


df_best_th = df.groupby("filename").apply(
    map_classes_best_percentage, threshold=best_thresholds
)
print(df_best_th.head())
df_best_th.reset_index().to_csv(
    "./analysis/processed_annotations_percentages/adapted_annotations_best_thresholds_paperVersion.csv",
    index=False,
)

exit()


### GET PERCENTILES BASED ON MEAN CATEGORY DURATION PER RECORDING

# Consider the mean duration per sample
mean_durations = df.groupby(["filename", "Superclass"])["duration"].mean().reset_index()
# print(mean_durations.head())
# print(mean_durations.shape)

mean_durations_by_class = {
    "Anth": mean_durations[mean_durations["Superclass"] == "Anthropophony"]["duration"],
    "Bio": mean_durations[mean_durations["Superclass"] == "Biophony"]["duration"],
    "Geo": mean_durations[mean_durations["Superclass"] == "Geophony"]["duration"],
    "Sil": mean_durations[mean_durations["Superclass"] == "Silence"]["duration"],
}

thresholds_mean = {
    p: {cls: np.percentile(dur, p) for cls, dur in mean_durations_by_class.items()}
    for p in percentiles
}
print("\nThresholds mean: ", thresholds_mean)


def map_classes_percentile_mean(x, threshold):
    anth = int(
        np.mean(x.loc[x["Superclass"] == "Anthropophony", "duration"])
        >= threshold["Anth"]
    )
    bio = int(
        np.mean(x.loc[x["Superclass"] == "Biophony", "duration"]) >= threshold["Bio"]
    )
    geo = int(
        np.mean(x.loc[x["Superclass"] == "Geophony", "duration"]) >= threshold["Geo"]
    )
    sil = int((anth + bio + geo == 0))
    return pd.Series(
        {
            "Anth": anth,
            "Bio": bio,
            "Geo": geo,
            "Sil": sil,
        }
    )


df_p5_mean = df.groupby("filename").apply(
    map_classes_percentile_mean, threshold=thresholds_mean[5]
)
df_p10_mean = df.groupby("filename").apply(
    map_classes_percentile_mean, threshold=thresholds_mean[10]
)
df_p25_mean = df.groupby("filename").apply(
    map_classes_percentile_mean, threshold=thresholds_mean[25]
)

# print("\n5%:")
# print(df_p5_mean.head(5))
# print("\n10%:")
# print(df_p10_mean.head())
# print("\n25%:")
# print(df_p25_mean.head())


### GET PERCENTILES BASED ON SUMMED CATEGORY DURATION PER RECORDING

# Consider the mean duration per sample
sum_durations = df.groupby(["filename", "Superclass"])["duration"].sum().reset_index()
print(sum_durations.head())
print(sum_durations.shape)

summed_durations_by_class = {
    "Anth": sum_durations[sum_durations["Superclass"] == "Anthropophony"]["duration"],
    "Bio": sum_durations[sum_durations["Superclass"] == "Biophony"]["duration"],
    "Geo": sum_durations[sum_durations["Superclass"] == "Geophony"]["duration"],
    "Sil": sum_durations[sum_durations["Superclass"] == "Silence"]["duration"],
}

thresholds_sum = {
    p: {cls: np.percentile(dur, p) for cls, dur in summed_durations_by_class.items()}
    for p in percentiles
}
print("\nSum thresholds: ", thresholds_sum)


def map_classes_percentile_sum(x, threshold):
    anth = int(
        np.sum(x.loc[x["Superclass"] == "Anthropophony", "duration"])
        >= threshold["Anth"]
    )
    bio = int(
        np.sum(x.loc[x["Superclass"] == "Biophony", "duration"]) >= threshold["Bio"]
    )
    geo = int(
        np.sum(x.loc[x["Superclass"] == "Geophony", "duration"]) >= threshold["Geo"]
    )
    sil = int((anth + bio + geo == 0))
    return pd.Series(
        {
            "Anth": anth,
            "Bio": bio,
            "Geo": geo,
            "Sil": sil,
        }
    )


df_p5_sum = df.groupby("filename").apply(
    map_classes_percentile_sum, threshold=thresholds_sum[5]
)
df_p10_sum = df.groupby("filename").apply(
    map_classes_percentile_sum, threshold=thresholds_sum[10]
)
df_p25_sum = df.groupby("filename").apply(
    map_classes_percentile_sum, threshold=thresholds_sum[25]
)

# print("\n5%:")
# print(df_p5_sum.head(5))
# print("\n10%:")
# print(df_p10_sum.head())
# print("\n25%:")
# print(df_p25_sum.head())


exit()


# Prepare annotations for new specified thresholds
def map_classes_adapted(x):
    anth = int(np.sum(x.loc[x["Superclass"] == "Anthropophony", "duration"]) >= 5)
    bio = int(np.sum(x.loc[x["Superclass"] == "Biophony", "duration"]) >= 2)
    geo = int(np.sum(x.loc[x["Superclass"] == "Geophony", "duration"]) >= 8)
    sil = int((anth + bio + geo == 0))
    return pd.Series(
        {
            "Anth": anth,
            "Bio": bio,
            "Geo": geo,
            "Sil": sil,
        }
    )


df_adapted_thresholds = df.groupby("filename").apply(map_classes_adapted)
print(df_adapted_thresholds.head())
