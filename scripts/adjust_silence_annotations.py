import os
import pandas as pd
from glob import glob

path = "/path/to/folder"

csv_filepaths = glob(os.path.join(path, "*.csv"))
for fp in csv_filepaths:
    print("\nFilepath: ", fp)
    df = pd.read_csv(fp)
    annotation_cols = ['Anth', 'Bio', 'Geo', 'Sil']
    df['annotation_sum'] = df[annotation_cols].sum(axis=1)
    condition = (df['Sil'] == 1) & (df['annotation_sum'] > 1)
    matching_entries = df[condition]
    print("Num entries: ", len(matching_entries))
    print("Entries: ", matching_entries)
    # Set those silence annotations to 0
    df.loc[condition, 'Sil'] = 0
    df.drop(columns='annotation_sum', inplace=True)
    print(df.loc[condition].head())
    df.to_csv(fp, index=False)
    # sys.exit()


