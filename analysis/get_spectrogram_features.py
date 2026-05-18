import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob


# --- Paths ---
features_path = "/path/to/log_mel_32k/segments"
spectrogram_paths = glob(os.path.join(features_path, "*.npy"))

dev_df = pd.read_csv("/path/to/dev.csv")
dev_df["index"] = dev_df.index 
print(dev_df.head())
print("dev_df shape: ", dev_df.shape)


features = []
for sp in spectrogram_paths:
    mel_spec = np.load(sp)
    mel_spec = mel_spec.squeeze() # Get rid of the channel

    # Compute features
    # energy
    energy = np.sum(mel_spec**2, axis=1)  # Shape: (1001,)
    # main (dominant) frequency
    main_mel_bin = np.argmax(mel_spec, axis=1)  # Shape: (1001,)
    # spectral centroid
    mel_bins = np.arange(mel_spec.shape[1])
    # Weighted average (centroid) for each frame
    spectral_centroid = np.sum(mel_spec * mel_bins, axis=1) / (np.sum(mel_spec, axis=1) + 1e-10)

    filename = os.path.join("segments", os.path.basename(sp).replace(".npy", ".wav"))

    # Aggregate per-spectrogram features
    features.append({
        "filename": filename,
        "energy_mean": energy.mean(),
        "energy_std": energy.std(),
        "main_mel_mean": main_mel_bin.mean(),
        "main_mel_std": main_mel_bin.std(),
        "spectral_centroid_mean": spectral_centroid.mean(),
        "spectral_centroid_std": spectral_centroid.std(),
    })

# Convert to DataFrame
features_df = pd.DataFrame(features)
print(features_df.head())
print(features_df.shape)

# Get intersections between the dev and features dataframes
matching_filenames = set(features_df["filename"]).intersection(set(dev_df["filename"]))
print("Number of matching files:", len(matching_filenames))
# Merge only relevant features
features_df = features_df.merge(dev_df[["filename", "index"]], on="filename", how="inner")
print(features_df.head(), features_df.shape)
# Reorder
cols = ["index", "filename", "energy_mean", "energy_std", "main_mel_mean", "main_mel_std",
        "spectral_centroid_mean", "spectral_centroid_std"]
features_df = features_df[cols]
print(features_df.head(), features_df.shape)
features_df.to_csv("./analysis/dev_spectrogram_features.csv", index=False)