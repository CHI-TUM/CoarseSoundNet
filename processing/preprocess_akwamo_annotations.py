import os
import pandas as pd

if __name__=='__main__':
    path = "/path/to/Akwamo_fixed/snippet_info_annotated.csv"
    df = pd.read_csv(path)
    print(df.head())
    print(df.shape)

    annotation_cols = [
        "Anthropophony_annotation",
        "Biophony_annotation",
        "Geophony_annotation",
        "Silence_annotation"
    ]
    
    df_filtered = df.dropna(subset=annotation_cols).reset_index(drop=True)
    df_filtered[annotation_cols] = df_filtered[annotation_cols].astype(int)
    print(df_filtered.shape)
    print(df_filtered.iloc[0, :])
    df_filtered = df_filtered[['snippet_filename'] + annotation_cols]
    df_filtered.rename(columns={"snippet_filename": "filename", "Anthropophony_annotation": "Anth", "Biophony_annotation": "Bio", "Geophony_annotation": "Geo", "Silence_annotation": "Sil"}, inplace=True)
    print(df_filtered.head())
    df_filtered.to_csv(path.replace(".csv", "_filtered.csv"), index=False)