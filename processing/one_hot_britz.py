import os
import pandas as pd
import numpy as np


def map_classes(x):
    anth = int(np.sum(x["Phony_Class"] == "Anthropophony") > 0)
    bio = int(np.sum(x["Phony_Class"] == "Biophony") > 0)
    geo = int(np.sum(x["Phony_Class"] == "Geophony") > 0)
    sil = int((anth + bio + geo == 0)) # Set Silence to 0 if any other annotation 
    return pd.Series({
        "Anthropophony": anth,
        "Biophony": bio,
        "Geophony": geo,
        "Silence": sil
    })

df = pd.read_csv("/path/to/Annotations_Britz_segments.csv")
superclasses = [
    "Anthropophony",
    "Biophony",
    "Geophony",
    "Silence",
]
df["Phony_Class"] = df["Phony_Class"].replace("Background", "Silence")
print("\nValueCount: ", df["Phony_Class"].value_counts())
df = df[df["Phony_Class"].isin(superclasses)]

print("\nSoundgroups: ", df["Soundgroup"].unique(), len(df["Soundgroup"].unique()))
print("Phony_Class: ", df["Phony_Class"].unique(), len(df["Phony_Class"].unique()))

# Do one-hot encoding
# Convert background to Silence, since we agreed on silence being completely silent without any other annotation
df = df.groupby("filename").apply(map_classes).reset_index()
df = df.set_index("filename")
print(df[superclasses].sum(0))

df = df.reset_index()
df = df.rename(columns={
    "Anthropophony": "Anth",
    "Biophony": "Bio",
    "Geophony": "Geo",
    "Silence": "Sil",
})
print(df)
df.to_csv("/path/to/Annotations_Britz_segments_one-hot.csv", index=False)

