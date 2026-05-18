
import os
import sys
import pandas as pd
import numpy as np
import argparse
from glob import glob


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True
    )
    args = parser.parse_args()

    predictions_path = args.path
    threshold = .4

    # Load and prepare predictions for new thresholds
    pred = pd.read_csv(predictions_path).set_index("filename")
    pred = pred.groupby("filename").apply(
        lambda x: pd.Series({
            "Anth": (x["Anth"] > .5).sum() >= 5,
            "Bio": (x["Bio"] > .5).sum() >= 2,
            "Geo": (x["Geo"] > .5).sum() >= 3,
            "Sil": (x["Sil"] > threshold).sum() >= 55,
        })
    )
    pred["Sil"] = pred.apply(lambda x: ~x["Anth"] & ~x["Bio"] & ~x["Geo"], axis=1)
    pred = pred.astype(int)
    pred.reset_index(inplace=True)
    print(pred.tail())
    pred.to_csv(predictions_path.split(".csv")[0] + "_postprocessed.csv", index=False)
