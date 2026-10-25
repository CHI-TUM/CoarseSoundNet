import os
import pandas as pd
import torch
import torchaudio
import torchlibrosa
import matplotlib.pyplot as plt

root = "/path/to/coarse/BE_data/original"

# -----------------------------
# Spectrogram params (same as before)
# -----------------------------
window_size_fft = 1024
hop_size = 320
sample_rate = 32000
fmin = 50
fmax = 14000
mel_bins = 64
ref = 1.0
amin = 1e-10
top_db = 80

WINDOW_SEC = 10.0      # model input window
FILE_DURATION = 60.0   # assumed total duration of each recording

spectrogram_extractor = torchlibrosa.stft.Spectrogram(
    n_fft=window_size_fft,
    win_length=window_size_fft,
    hop_length=hop_size,
)

logmel_extractor = torchlibrosa.stft.LogmelFilterBank(
    sr=sample_rate,
    fmin=fmin,
    fmax=fmax,
    n_mels=mel_bins,
    n_fft=window_size_fft,
    ref=ref,
    amin=amin,
    top_db=top_db,
)


def compute_logmel(path: str, start_sec: float, end_sec: float):
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
        sr = sample_rate

    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)

    segment = waveform[:, start_sample:end_sample]
    segment = segment.squeeze(0).unsqueeze(0)

    with torch.no_grad():
        spec = spectrogram_extractor(segment)
        logmel = logmel_extractor(spec).squeeze(1)
        logmel_np = logmel[0].cpu().numpy()

    return logmel_np


def get_window(begin: float, end: float,
                total_duration: float = FILE_DURATION,
                window_size: float = WINDOW_SEC):
    """
    Return (start, end) of a `window_size`-second window covering the event,
    centered on the event midpoint and clamped to file bounds.
    If the event spans (almost) the whole file, fall back to a fixed
    representative window (20-30s).
    """
    event_duration = end - begin

    if event_duration >= total_duration - 1e-6:
        start = min(20.0, max(0.0, total_duration - window_size))
        return start, start + window_size

    center = (begin + end) / 2.0
    start = center - window_size / 2.0
    stop = start + window_size

    if start < 0:
        start, stop = 0.0, window_size
    if stop > total_duration:
        stop = total_duration
        start = total_duration - window_size

    return start, stop


# -----------------------------
# Load group CSVs
# -----------------------------
g1 = pd.read_csv("./analysis/error_analysis_filenames/group1_anth_overlap_timespans.csv")
g2 = pd.read_csv("./analysis/error_analysis_filenames/group2_geo_timespans.csv")
g3 = pd.read_csv("./analysis/error_analysis_filenames/group3_insect_orthoptera_timespans.csv")

# # def pick_top_examples(df, begin_col, end_col, n=3, max_duration=30.0, min_duration=1.0):
# def pick_top_examples(df, begin_col, end_col, n=3):
#     """
#     Pick the n longest-duration events, keeping at most one event per file
#     (so the 3 examples come from 3 different recordings).
#     """
#     df = df.copy()
#     df["duration"] = df[end_col] - df[begin_col]
#     # df = df[(df["duration"] >= min_duration) & (df["duration"] <= max_duration)]  
#     df = df.sort_values("duration", ascending=False)
#     df = df.drop_duplicates(subset="filename", keep="first")
#     return df.head(n)


# # Group 1: use the overlap window itself (anth masked by bio/geo)
# g1_examples = pick_top_examples(g1, "overlap_start_s", "overlap_end_s", n=3)

# # Group 2: geophony segments
# g2_examples = pick_top_examples(g2, "Begin Time (s)", "End Time (s)", n=3)

# # Group 3: insect/orthoptera segments
# g3_examples = pick_top_examples(g3, "Begin Time (s)", "End Time (s)", n=3)


# -----------------------------
# Manually choose files per group
# -----------------------------

# Group 1: specify the filenames you want (no .wav extension, matches "filename" column)
group1_choices = [
    "0420-14042016_6", # Bio #"2110-19072016_212",
    "1430-12032016_306", #"1430-14032016_139", # "1430-02052016_104", # "1310-24052016_192", # Geo # "1340-22032016_324",
    "0800-29032016_324", # Bio + Geo #"2000-22072016_210",
]
group1_titles = ["Bio", "Geo", "Bio + Geo"]

# Group 2
group2_choices = [
    "0000-05072016_4", # Rain
    "0700-22032016_199", # Wind # "2130-24062016_278"
    "2050-12072016_13", # Geo
]
group2_titles = ["Rain", "Wind", "Geo"]

# Group 3
group3_choices = [
    "2350-17072016_198",
    "2130-05072016_148",
    "2250-23072016_233",
]
group3_titles = ["Insect", "Insect", "Insect"]

title_dict = {
    0: group1_titles,
    1: group2_titles,
    2: group3_titles,
}

def pick_chosen_examples(df, filenames, begin_col, end_col, max_duration=60.0, min_duration=1.0):
    """
    For each filename in `filenames`, pick one event row from df:
    the longest event within [min_duration, max_duration], in the given order.
    Raises if a filename has no valid row, so you notice immediately.
    """
    df = df.copy()
    df["duration"] = df[end_col] - df[begin_col]
    valid = df[(df["duration"] >= min_duration) & (df["duration"] <= max_duration)]

    rows = []
    for fname in filenames:
        sub = valid[valid["filename"] == fname]
        if sub.empty:
            raise ValueError(f"No valid event found for '{fname}' in this group's CSV "
                              f"(check filename spelling or duration bounds).")
        # pick the longest qualifying event for this file
        best = sub.sort_values("duration", ascending=False).iloc[0]
        rows.append(best)

    return pd.DataFrame(rows)

g1_examples = pick_chosen_examples(g1, group1_choices, "overlap_start_s", "overlap_end_s")
g2_examples = pick_chosen_examples(g2, group2_choices, "Begin Time (s)", "End Time (s)")
g3_examples = pick_chosen_examples(g3, group3_choices, "Begin Time (s)", "End Time (s)")


print("Group 1 selected examples:\n", g1_examples[["filename", "overlap_start_s", "overlap_end_s", "other_superclass", "other_annotation"]])
print("\nGroup 2 selected examples:\n", g2_examples[["filename", "Begin Time (s)", "End Time (s)", "Annotation"]])
print("\nGroup 3 selected examples:\n", g3_examples[["filename", "Begin Time (s)", "End Time (s)", "Annotation"]])

SUPERCLASS_ABBR = {
    "Anthropophony": "Anth",
    "Biophony": "Bio",
    "Geophony": "Geo",
    "Silence": "Sil",
}

def abbr(name: str) -> str:
    """Map a full superclass name to its abbreviation; pass through unknown values."""
    return SUPERCLASS_ABBR.get(str(name).strip(), str(name).strip())

# -----------------------------
# Build (label, path, start, end) entries per group
# -----------------------------
def build_entries_group1(df):
    entries = []
    for _, row in df.iterrows():
        start, end = get_window(row["overlap_start_s"], row["overlap_end_s"])
        label = f"{row['filename']}\nAnth + {abbr(row['other_superclass'])} ({abbr(row['other_annotation'])})"
        # label = f"Anth + {abbr(row['other_superclass'])} ({abbr(row['other_annotation'])})"
        path = os.path.join(root, row["filename"] + ".wav")
        entries.append((label, path, start, end))
    return entries


def build_entries_generic(df, begin_col, end_col, ann_col):
    entries = []
    for _, row in df.iterrows():
        start, end = get_window(row[begin_col], row[end_col])
        label = f"{row['filename']}\n{abbr(row[ann_col])}"
        # label = f"{abbr(row[ann_col])}"
        path = os.path.join(root, row["filename"] + ".wav")
        entries.append((label, path, start, end))
    return entries


group_entries = {
    "Anth missed with Bio/Geo present": build_entries_group1(g1_examples), # Group 1: 
    "Geo interpreted as Sil": build_entries_generic(g2_examples, "Begin Time (s)", "End Time (s)", "Annotation"), # Group 2: 
    "Insect (Bio) missed": build_entries_generic(g3_examples, "Begin Time (s)", "End Time (s)", "Annotation"), # Group 3: Insect / Orthoptera
}

# -----------------------------
# Compute spectrograms
# -----------------------------
spectrograms = {}
for group_name, entries in group_entries.items():
    for label, path, start, end in entries:
        spectrograms[(group_name, label)] = compute_logmel(path, start, end)

# -----------------------------
# Plot: one row per group, 3 examples per row
# -----------------------------
n_groups = len(group_entries)
n_examples = 3

# fig = plt.figure(figsize=(15, 4 * n_groups))
# gs = fig.add_gridspec(n_groups, n_examples + 1,
#                        width_ratios=[1, 1, 1, 0.05], wspace=0.15, hspace=0.5)

# im = None
# for row_idx, (group_name, entries) in enumerate(group_entries.items()):
#     for col_idx, (label, path, start, end) in enumerate(entries):
#         ax = fig.add_subplot(gs[row_idx, col_idx])
#         spec = spectrograms[(group_name, label)]
#         im = ax.imshow(
#             spec.T,
#             origin="lower",
#             aspect="auto",
#             interpolation="nearest"
#         )
#         ax.set_title(label, fontsize=9)
#         ax.set_xlabel(f"Time ({end - start:.0f}s window)")
#         ax.set_xticks([])
#         if col_idx == 0:
#             ax.set_ylabel(f"{group_name}\nMel bins", fontsize=9)

#     cax = fig.add_subplot(gs[row_idx, n_examples])
#     fig.colorbar(im, cax=cax, label="Log-Mel energy")

# plt.tight_layout()


fig = plt.figure(figsize=(15, 4.5 * n_groups), constrained_layout=True)

# Two rows per group: a slim title row, then the plot row
gs = fig.add_gridspec(
    n_groups * 2, n_examples + 1,
    width_ratios=[1, 1, 1, 0.05],
    height_ratios=[0.12, 1] * n_groups,  # title row is much shorter than plot row
)

im = None
for row_idx, (group_name, entries) in enumerate(group_entries.items()):
    # print("\nGroup name: ", group_name)
    # print("Row idx: ", row_idx)
    title_row = row_idx * 2
    plot_row = row_idx * 2 + 1

    # Dedicated title axis, spanning the plot columns (not the colorbar column)
    ax_title = fig.add_subplot(gs[title_row, :n_examples])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5, group_name, ha="center", va="center",
                   fontsize=15, fontweight="bold")

    for col_idx, (label, path, start, end) in enumerate(entries):
        # print("Col_idx: ", col_idx)
        ax = fig.add_subplot(gs[plot_row, col_idx])
        spec = spectrograms[(group_name, label)]
        im = ax.imshow(
            spec.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest"
        )
        # label_woT = label.split("\n")[-1]
        label_manual = title_dict[row_idx][col_idx]
        ax.set_title(label_manual, fontsize=14)
        ax.set_xlabel(f"Time ({end - start:.0f}s window)")
        ax.set_xticks([])
        if col_idx == 0:
            ax.set_ylabel("Mel bins", fontsize=14)

    cax = fig.add_subplot(gs[plot_row, n_examples])
    fig.colorbar(im, cax=cax, label="Log-Mel energy")


plt.savefig("./analysis/error_analysis_filenames/error_analysis_spectrograms.pdf")
plt.show()