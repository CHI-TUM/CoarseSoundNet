import os
import pandas as pd

# Load the CSV file
df = pd.read_csv("annotations_BEForest.csv")  # adjust path if needed
df["file"] = df["file"] + ".wav"
df = df.rename(columns={"file": "filename"})

# Define mapping for shorter names
superclass_rename_map = {
    "Anthropophony": "Anth",
    "Biophony": "Bio",
    "Geophony": "Geo",
    "Silence": "Sil"
}

# Group by 'file' and 'Superclass', summing the durations
summary = df.groupby(['filename', 'Superclass'])['Duration'].sum().unstack(fill_value=0)
# summary = summary.rename(columns=lambda col: f"{superclass_rename_map.get(col, col)}_duration")
summary = summary.rename(columns={col: f"{superclass_rename_map.get(col, col)}_duration" for col in summary.columns if col != summary.index.name})
summary = summary.sort_index()

# Also add insect duration
insect_labels = ["Insect", "Gryllus campestris", "Roeseliana roeselii", "Orthoptera"]
insect_summary = df[df["Annotation"].isin(insect_labels)].groupby("filename")["Duration"].sum()
insect_summary = insect_summary.reindex(summary.index, fill_value=0)
summary["Insect_duration"] = insect_summary
print(summary.head(), summary.shape)

combined_df = pd.read_csv("/path/to/predictions/reviewed_annotations/edansa_hf/combined_pred_prob_gt.csv")
combined_df = combined_df.set_index("filename")
print(combined_df.head(), combined_df.shape)

combined_df = pd.concat([summary, combined_df], axis=1).reset_index()
print(combined_df.head(10))
print(combined_df.shape)

# Save the result
combined_df.to_csv("/path/to/predictions/reviewed_annotations/edansa_hf/superclass_probs_duration_summary.csv", index=False)


# Combine indices and combined_df
df_dom = pd.read_csv("/path/to/acoustic_indices_soundecology.csv")
df_dom = df_dom[['file', 'ACI', 'ADI', 'NDSI']]
df_dom = df_dom.rename(columns={'file':'filename'})
df_dom.set_index("filename", inplace=True)

combined_df.set_index("filename", inplace=True)
common_index = combined_df.index.intersection(df_dom.index)
combined_df = combined_df.loc[common_index]
combined_df = combined_df.sort_index()
df_dom = df_dom.loc[common_index]
df_dom = df_dom.sort_index()
df_combined = pd.concat([df_dom, combined_df], axis=1).reset_index()
df_combined.to_csv("/path/to/BESound_indices_summary.csv", index=False)