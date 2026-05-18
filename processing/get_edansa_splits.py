import os
import pandas as pd

if __name__=='__main__':
    path = "/path/to/EDANSA-2019"

    df = pd.read_csv(os.path.join(path, "labels.csv"))
    df.rename(columns={'Clip Path': 'filename'}, inplace=True)
    target_columns = df.columns[9:]
    for col in target_columns:
        df[col] = df[col].astype(float)
    df_train = df.loc[df["set"] == "train"]
    df_dev = df.loc[df["set"] == "valid"]
    df_test = df.loc[df["set"] == "test"]
    df_train.to_csv(os.path.join(path, "train.csv"), index=False)
    df_dev.to_csv(os.path.join(path, "dev.csv"), index=False)
    df_test.to_csv(os.path.join(path, "test.csv"), index=False)
    print(df_train.head())
    print("Done.")