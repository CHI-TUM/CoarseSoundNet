import argparse
import glob
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    SPLIT_RATIO = 0.85
    random.seed(42)

    files = glob.glob(
        os.path.join(
            args.src,
            "**",
            "*.wav"
        ),
        recursive=True
    )
    # Shuffle files
    random.shuffle(files)

    # 2. Split into train and test
    split_idx = int(len(files) * SPLIT_RATIO)
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    # Create DataFrames
    df_train = pd.DataFrame({
        "file": train_files
    })
    df_train["start"] = None
    df_train["end"] = None
    df_train["split"] = "train"

    df_test = pd.DataFrame({
        "file": test_files
    })
    df_test["start"] = None
    df_test["end"] = None
    df_test["split"] = "test"

    # Combine into one DataFrame
    df = pd.concat([df_train, df_test], ignore_index=True)
    df["meta"] = "Wind"
    print(df, df.shape)

    df.to_csv(os.path.join(args.dst, "bbc_wind.csv"), index=False)