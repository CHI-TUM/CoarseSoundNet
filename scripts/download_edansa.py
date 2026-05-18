"""
Download the EDANSA dataset. Taken from https://autrainer.github.io/autrainer/_modules/autrainer/datasets/edansa2019.html#EDANSA2019
"""
import pandas as pd
import os
import shutil
from argparse import ArgumentParser
from autrainer.datasets.utils import ZipDownloadManager

FILES = {
    "EDANSA-2019.zip": "https://zenodo.org/records/6824272/files/EDANSA-2019.zip?download=1"
}

if __name__ == "__main__":
    # parser = ArgumentParser("--- Download Edansa ---")
    # parser.add_argument()
    path = "/nas/staff/data_work/AG/HearTheSpecies"

    out_path = os.path.join(path, "default")
    os.makedirs(out_path, exist_ok=True)

    # download and extract files
    dl_manager = ZipDownloadManager(FILES, path)
    dl_manager.download(check_exist=["EDANSA-2019"])
    dl_manager.extract(check_exist=["EDANSA-2019"])

    # move audio files
    for item in os.listdir(os.path.join(path, "EDANSA-2019", "data")):
        shutil.move(
            os.path.join(path, "EDANSA-2019", "data", item),
            os.path.join(out_path, item),
        )

    # process dataframes
    df = pd.read_csv(os.path.join(path, "EDANSA-2019", "labels.csv"))
    target_columns = df.columns[9:]
    for col in target_columns:
        df[col] = df[col].astype(float)
    df_train = df.loc[df["set"] == "train"]
    df_dev = df.loc[df["set"] == "valid"]
    df_test = df.loc[df["set"] == "test"]
    df_train.to_csv(os.path.join(path, "train.csv"), index=False)
    df_dev.to_csv(os.path.join(path, "dev.csv"), index=False)
    df_test.to_csv(os.path.join(path, "test.csv"), index=False)

    # remove unnecessary files
    shutil.rmtree(os.path.join(path, "EDANSA-2019"))