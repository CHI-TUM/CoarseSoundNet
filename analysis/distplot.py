import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob

# === Load dev.csv ===
df = pd.read_csv("/path/to/coarse/BE_data/dev.csv")

# === Load all spectrogram paths ===
features_path = "/path/to/coarse/BE_data/log_mel_32k/segments"
spectrogram_paths = {os.path.join("segments", os.path.basename(p).replace(".npy", ".wav")): p for p in glob(os.path.join(features_path, "*.npy"))}


# === Initialize energy bins per class ===
class_names = ["Anth", "Bio", "Geo", "Sil"]
mel_bin_energies = {cls: [] for cls in class_names}

# === Loop through all files ===
for _, row in df.iterrows():
    fname = row["filename"]
    if fname not in spectrogram_paths:
        continue

    mel_spec = np.load(spectrogram_paths[fname]).squeeze()  # (time, mel_bins)
    energy_per_bin = np.mean(mel_spec**2, axis=0)  # shape: (64,)
    
    for cls in class_names:
        if row[cls] == 1:
            mel_bin_energies[cls].append(energy_per_bin)

# === Convert to dB and average ===
plt.figure(figsize=(10, 6))
for cls, energies in mel_bin_energies.items():
    if not energies:
        continue
    avg_energy = np.mean(energies, axis=0)  # shape: (64,)
    # avg_db = 10 * np.log10(avg_energy + 1e-10)  # convert to dB scale
    avg_db = avg_energy
    plt.plot(avg_db, label=cls)

plt.xlabel("Mel Bin")
plt.ylabel("Energy")
plt.title("Average Energy per Mel Bin for Each Class")
plt.legend(title="Label")
plt.grid(True)
plt.tight_layout()
plt.savefig("analysis/distplot.png", dpi=300)
plt.close()
