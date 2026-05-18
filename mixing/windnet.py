import argparse
import glob
import os
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    df = pd.read_csv("/path/to/WindNet/wind-noise-detection-main/data/annotation_5sec/annotations_SA_clean_final.csv")
    # Filter for only wind or rain to make it only Geophony (can be adjusted for a more sophisticated mixing later on, where multiple active annotations are possible)
    df = df[((df["wind"] == 1) | (df["rain"] == 1)) & (df["animal_sound"] == 0)]
    # print(df)
    df["file"] = "/path/to/WindNet/wind-noise-detection-main/data/208_file_recordings_paper_wind/" + df["file_name"]
    df["start"] = df["segment_start_s"]
    df["end"] = df["segment_start_s"] + 5.0
    df["meta"] = "Geo"
    df.reset_index(drop=True, inplace=True)
    
    # Assign train and test splits
    split_idx = int(len(df) * 0.85)
    df["split"] = "train"
    df.loc[split_idx:, "split"] = "test"
    print("Train num: ", len(df[df["split"] == "train"]))
    print("Test num: ", len(df[df["split"] == "test"]))

    # Prepare and store as csv
    df = df[["file", "start", "end", "split", "meta"]]
    df.to_csv(os.path.join(args.dst, "windnet.csv"), index=False)
    