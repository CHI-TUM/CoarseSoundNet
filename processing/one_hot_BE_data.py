import os
import pandas as pd
import numpy as np

def map_classes(x):
    anth = int(np.sum(x["Superclass"] == "Anthropophony") > 0)
    bio = int(np.sum(x["Superclass"] == "Biophony") > 0)
    geo = int(np.sum(x["Superclass"] == "Geophony") > 0)
    sil = int((anth + bio + geo == 0)) # Set Silence to 0 if any other annotation 
    return pd.Series({
        "Anthropophony": anth,
        "Biophony": bio,
        "Geophony": geo,
        "Silence": sil
    })

df = pd.read_csv("/path/to/aggregated_tables_segments_10s.csv")
superclasses = [
    "Anthropophony",
    "Biophony",
    "Geophony",
    "Silence",
]
print(df.head())
df.rename(columns={"file": "filename"}, inplace=True)
df = df.groupby("filename").apply(map_classes).reset_index()

df = df.rename(columns={
    "Anthropophony": "Anth",
    "Biophony": "Bio",
    "Geophony": "Geo",
    "Silence": "Sil",
})
print(df)

df.to_csv("/path/to/store/Annotations_BEForest_segments_one-hot.csv", index=False)