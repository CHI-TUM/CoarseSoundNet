"""Just temporary until we figured out the issue with the filepaths in the new autrainer version.
Taken from the branch iid.
"""

import pandas as pd
import os
import glob

def _mixed_df(split: str = "train") -> pd.DataFrame:
    seed = 42
    data = {
        "filename": [],
        "A": [],
        "B": [],
        "G": [],
        "S": [],
    }
    path = "train" if split == "dev" else split
    for cat in ["A", "B", "G", "AB", "AG", "BG", "ABG"]:
        cat_files = glob.glob(
            f"/data/chi-gpu4/gebhaale/synthetic_data/{path}/{cat}/*.wav"
        )
        data["filename"] += cat_files
        for l in ["A", "B", "G"]:
            if l in cat:
                data[l] += [1] * len(cat_files)
            else:
                data[l] += [0] * len(cat_files)
        data["S"] += [0] * len(cat_files)
    silence = sorted(
        glob.glob("/data/chi-gpu4/gebhaale/synthetic_data/silence/*.wav")
    )
    if split in ["train", "dev"]:
        silence = silence[
            :8500
        ]  # we have 8500 train/dev files and 1500 test files for the categories A, B, G, etc.
    elif split == "test":
        silence = silence[8500:]
    df = pd.DataFrame(data)
    sil_df = pd.DataFrame(data=silence, columns=["filename"])
    sil_df["A"] = 0
    sil_df["B"] = 0
    sil_df["G"] = 0
    sil_df["S"] = 1
    df = pd.concat((df, sil_df))
    df = df.reset_index(drop=True)
    df = df.rename(columns={"A": "Anth", "B": "Bio", "G": "Geo", "S": "Sil"})
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # split into train/dev
    if split == "train":
        df = df[:round(len(df)*.85)]
    elif split == "dev":
        df = df[round(len(df)*.85):]

    df["filename"] = df["filename"].apply(lambda x: x.replace("/data/chi-gpu4/gebhaale/synthetic_data/", ""))
    return df


if __name__=='__main__':
    df_train = _mixed_df("train")
    df_dev = _mixed_df("dev")
    df_test = _mixed_df("test")

    print("train:\n", df_train.head(), df_train.shape)
    print("dev:\n", df_dev.head(), df_dev.shape)
    print("test:\n", df_test.head(), df_test.shape)

    df_train.to_csv("/data/chi-gpu4/gebhaale/synthetic_data/train.csv", index=False)
    df_dev.to_csv("/data/chi-gpu4/gebhaale/synthetic_data/dev.csv", index=False)
    df_test.to_csv("/data/chi-gpu4/gebhaale/synthetic_data/test.csv", index=False)