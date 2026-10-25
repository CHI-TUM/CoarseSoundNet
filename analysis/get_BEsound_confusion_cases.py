import pandas as pd
import os

"""
The three groups were calculated based on evaluate_best_percentile_annotations.py with the best PDA and CST.
This script aims to find the specific time intervals of the classes missed or confused for each filename.
"""

# ---------- Load inputs ----------

fine = pd.read_csv("./annotations_BEForest.csv")

# Strip whitespace on the label columns just in case
fine["Superclass"] = fine["Superclass"].astype(str).str.strip()
fine["Annotation"] = fine["Annotation"].astype(str).str.strip()

group1_files = pd.read_csv("./analysis/error_analysis_filenames/group1_anth_masked.csv")["filename"].tolist()
group2_files = pd.read_csv("./analysis/error_analysis_filenames/group2_geo_confused_with_silence.csv")["filename"].tolist()
group3_files = pd.read_csv("./analysis/error_analysis_filenames/group3_bio_missed.csv")["filename"].tolist()

# Group CSVs have ".wav" suffix, fine-grained "file" column doesn't -> strip suffix for matching
def strip_wav(f):
    return os.path.splitext(f)[0]

group1_stems = [strip_wav(f) for f in group1_files]
group2_stems = [strip_wav(f) for f in group2_files]
group3_stems = [strip_wav(f) for f in group3_files]


# ---------- Helper: interval overlap ----------

def interval_overlaps(a_start, a_end, b_start, b_end):
    """Return (overlap_start, overlap_end) if intervals overlap, else None."""
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if start < end:
        return start, end
    return None


# ---------- Group 1: Anthropophony overlapping Bio/Geo ----------

g1_rows = []
fine_g1 = fine[fine["file"].isin(group1_stems)]

for fname, sub in fine_g1.groupby("file"):
    anth_segs = sub[sub["Superclass"] == "Anthropophony"]
    other_segs = sub[sub["Superclass"].isin(["Biophony", "Geophony"])]

    for _, a in anth_segs.iterrows():
        for _, o in other_segs.iterrows():
            ov = interval_overlaps(
                a["Begin Time (s)"], a["End Time (s)"],
                o["Begin Time (s)"], o["End Time (s)"]
            )
            if ov is not None:
                g1_rows.append({
                    "filename": fname,
                    "overlap_start_s": ov[0],
                    "overlap_end_s": ov[1],
                    "anth_annotation": a["Annotation"],
                    "anth_start_s": a["Begin Time (s)"],
                    "anth_end_s": a["End Time (s)"],
                    "other_superclass": o["Superclass"],
                    "other_annotation": o["Annotation"],
                    "other_start_s": o["Begin Time (s)"],
                    "other_end_s": o["End Time (s)"],
                })

g1_df = pd.DataFrame(g1_rows).sort_values(["filename", "overlap_start_s"])
g1_df.to_csv("./analysis/error_analysis_filenames/group1_anth_overlap_timespans.csv", index=False)
print(f"Group 1: {len(g1_df)} overlapping time spans across {g1_df['filename'].nunique()} files")


# ---------- Group 2: Geophony time spans ----------

fine_g2 = fine[fine["file"].isin(group2_stems) & (fine["Superclass"] == "Geophony")]

g2_df = fine_g2[[
    "file", "Annotation", "Begin Time (s)", "End Time (s)", "Duration"
]].rename(columns={"file": "filename"}).sort_values(["filename", "Begin Time (s)"])

g2_df.to_csv("./analysis/error_analysis_filenames/group2_geo_timespans.csv", index=False)
print(f"Group 2: {len(g2_df)} Geophony time spans across {g2_df['filename'].nunique()} files")


# ---------- Group 3: Insect / Orthoptera time spans ----------

insect_mask = fine["Annotation"].str.contains("insect", case=False, na=False) | \
              fine["Annotation"].str.contains("orthoptera", case=False, na=False)

fine_g3 = fine[fine["file"].isin(group3_stems) & insect_mask]

g3_df = fine_g3[[
    "file", "Annotation", "Begin Time (s)", "End Time (s)", "Duration"
]].rename(columns={"file": "filename"}).sort_values(["filename", "Begin Time (s)"])

g3_df.to_csv("./analysis/error_analysis_filenames/group3_insect_orthoptera_timespans.csv", index=False)
print(f"Group 3: {len(g3_df)} Insect/Orthoptera time spans across {g3_df['filename'].nunique()} files")

# ---------- Sanity checks: files in group list but with no matching spans ----------

for label, stems, result_df in [
    ("Group 1", group1_stems, g1_df),
    ("Group 2", group2_stems, g2_df),
    ("Group 3", group3_stems, g3_df),
]:
    missing = set(stems) - set(result_df["filename"].unique())
    if missing:
        print(f"\n{label}: {len(missing)}/{len(stems)} files had no matching time spans in the fine-grained file:")
        for m in sorted(missing):
            print(f"  {m}")