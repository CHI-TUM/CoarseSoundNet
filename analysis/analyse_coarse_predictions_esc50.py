import os
import pandas as pd
import numpy as np
import seaborn as sns
import glob
import tqdm
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score

# Group by 'filename' and count thresholded offsets per soundscape, then filter
def check_majority(group, target_columns) -> pd.Series:
    majority_count = len(group) / 2
    present_soundscapes = [col for col in target_columns if group[col].sum() >= majority_count]
    return pd.Series({'present_soundscapes': present_soundscapes})


def load_annotations(
    fp: str = "./annotations.csv", 
    root: str = "/path/to/BEforest/files"
) -> pd.DataFrame:

    if os.path.exists(fp):
        df = pd.read_csv(fp)
        # df.rename(columns={'file': 'filename'}, inplace=True)
        # df["filename"] += ".wav"
    else:
        files = glob.glob(os.path.join(root, "*.txt"))
        df = pd.DataFrame()

        for file in tqdm.tqdm(files, total=len(files), desc="Read"):
            local_df = pd.read_csv(file, sep="\t")
            local_df["file"] = os.path.basename(file).split(".")[0]
            df = pd.concat((df, local_df), ignore_index=True)
        # df.to_csv(args.dest, index=False)

        # per file
        print(df.head(5))
        df = df.set_index(["file", "Selection"])
        df = df.loc[~df.index.duplicated()]
        df = df.reset_index()

        df["Plot"] = df["file"].apply(lambda x: x.split("_")[1])
        df["Date"] = df["file"].apply(lambda x: x.split("_")[0].split("-")[1])


        # Manual fixes
        df.loc[(df["file"] == "0220-06072016_216") & (df["Annotation"] == "Noise"), "Annotation"] = "Rain"
        df.loc[(df["file"] == "0240-05042016_242") & (df["Annotation"] == "Noise"), "Annotation"] = "Silence"
        df.loc[(df["file"] == "0230-27062016_122") & (df["Annotation"] == "Noise"), "Annotation"] = "Wind"
        df.loc[(df["file"] == "0310-16072016_308") & (df["Annotation"] == "Noise"), "Annotation"] = "Wind"

        anthropophony = [
            "Airplane",
            "Anthropohony",
            "Anthropophony",
            "Car",
            "Chainsaw",
            "Church bell",
            "Engine noise",
            "Helicopter",
            "Human"
        ]

        geophony = [
            "Geophony",
            "Geophpny",
            "Geophpony",
            "Rain",
            "Thunder",
            "Wind"
        ]

        silence = [
            "Silence"
        ]
        exclude = [
            "Other",
            "Interference",
        ]

        df["Duration"] = df["End Time (s)"] - df["Begin Time (s)"]
        df.loc[(df["file"] == "0030-14062016_102") & (df["Annotation"] == "Silence"), "Annotation"] = None
        df.loc[(df["file"] == "0310-23072016_316") & (df["Annotation"] == "Small mammal"), "Annotation"] = None
        df.loc[(df["file"] == "0020-28042016_31") & (df["Annotation"] == "Insect"), "Annotation"] = None
        df = df.loc[~df["Annotation"].isna()]

        silent_files = df.loc[df["Annotation"] == "Silence", "file"].values

        df = df.loc[~df["Annotation"].isin(exclude)]
        df["Superclass"] = "Biophony"
        df.loc[df["Annotation"].isin(geophony), "Superclass"] = "Geophony"
        df.loc[df["Annotation"].isin(anthropophony), "Superclass"] = "Anthropophony"
        df.loc[df["Annotation"] == "Silence", "Superclass"] = "Silence"
        df["Hours"] = df["Duration"] / 3600

        print(df.head())
        df.to_csv(fp, index=False)

    return df

def map_classes(x):
    anth = int(np.sum(x["Superclass"] == "Anthropophony") > 0)
    bio = int(np.sum(x["Superclass"] == "Biophony") > 0)
    geo = int(np.sum(x["Superclass"] == "Geophony") > 0)
    sil = int((anth + bio + geo == 0))
    return pd.Series({
        "Anthropophony": anth,
        "Biophony": bio,
        "Geophony": geo,
        "Silence": sil
    })

def map_time(x):
    if len(x) == 2:
        h = 0
        m = int(x[:2])
    else:
        h = int(x[:2])
        m = int(x[2:])
    if m >= 30:
        h += 1
    if h >= 24:
        h -= 24
    return h


if __name__=='__main__':
    predictions_path = "/path/to/predictions"
    variant = "max"

    superclasses = [
        "Anthropophony",
        "Biophony",
        "Geophony"
    ]

    meta_path = "/path/to/meta.csv"
    df = pd.read_csv(meta_path)

    df.rename(
        columns={
            "mix_id": "file", 
            "filename": "file", 
            "supercategory": "Superclass",
            "Anth": "Anthropophony",
            "Bio": "Biophony",
            "Geo": "Geophony",
            "Sil": "Silence"
        }, 
        inplace=True
    )
    if "mix" in meta_path:
        df["file"] = df["file"].apply(lambda x: "mixings_wav/{:012d}.wav".format(x))
    # df = load_annotations(fp="./annotations_BEForest.csv")
    # df["file"] += ".wav"

    # df_file = df.groupby("file").apply(map_classes).reset_index()
    df_file = df[["file"] + superclasses]
    # df_file["Time"] = df_file["file"].apply(lambda x: map_time(x.split("-")[0]))
    df_file = df_file.set_index("file")
    print(df_file.head(), df_file.shape)
    

    # Load the predictions
    pred = pd.read_csv(os.path.join(predictions_path, "results.csv"))
    pred.rename(columns={"filename": "file"}, inplace=True)
    pred = pred.rename(columns={
        "Anth": "Anthropophony",
        "Bio": "Biophony",
        "Geo": "Geophony",
        "Sil": "Silence",
    })
    pred = pred.drop(columns=["prediction", "output"])
    
    # individual thresholds
    thresholds = {
        "Anthropophony": 0.5,
        "Biophony": 0.5,
        "Geophony": 0.5,
        "Silence": 0.5
    }
    
    # if max
    if variant == "max":
        if "offset" in pred.columns:
            pred.drop(columns=["offset"], inplace=True)
        pred = pred.groupby("file").max()
        pred = pred.apply(lambda col: (col > thresholds[col.name]).astype(int))
    # if sliding window with majority voting
    elif variant == "majority":
        df_majority_own = pred.copy()
        df_majority_own = df_majority_own[df_majority_own['offset'] != 'majority']
        df_majority_own.drop(columns=["offset"], inplace=True)
        # Apply column-specific thresholding
        for col in superclasses:
            df_majority_own[col] = df_majority_own[col] > thresholds[col]
        df_majority_own = df_majority_own.groupby('file').apply(lambda x: check_majority(x, superclasses)).reset_index()
        # df_majority_own['soundscape_prediction'] = [0 if len(x) == 0 else 1 for x in df_majority_own['present_soundscapes']]
        pred = df_majority_own

        def update_columns(row):
            row_dict = {col: 1 if col in row["present_soundscapes"] else 0 for col in superclasses}
            
            # If present_soundscapes is empty, set Silence to 1
            if not row["present_soundscapes"]:  
                row_dict["Silence"] = 1
            
            return pd.Series(row_dict)
        # pred[superclasses] = pred.apply(lambda row: [1 if col in row["present_soundscapes"] else 0 for col in superclasses], axis=1, result_type="expand")
        pred[superclasses] = pred.apply(update_columns, axis=1)
        pred.drop(columns=["present_soundscapes"], inplace=True)
        pred = pred.set_index("file")
    else:
        pred = pred.set_index("file")
        pred = pred.apply(lambda col: (col > thresholds[col.name]).astype(int))
    pred = pred.replace(2, 1)



    # Align indices
    common_index = df_file.index.intersection(pred.index)
    df_file = df_file.loc[common_index]
    df_file = df_file.sort_index()
    pred = pred.loc[common_index]
    pred = pred.sort_index()

    # Make sure the indices are the same
    assert len(set(pred.index).intersection(set(df_file.index))) == len(df_file)
    assert len(pred.index) == len(df_file.index)

    
    


    # pred["Silence"] = pred.apply(lambda x: int((x["Silence"] & x.sum() == 1) | (x.sum() == 0)), axis=1)
    print(pred.head(), pred.shape)
    # print(df_file, pred)
    # pred = pred.drop(columns=["Silence"])
    # df_file = df_file.drop(columns=["Silence"])
    print(df_file[superclasses].sum())


    print(f1_score(df_file["Anthropophony"], pred["Anthropophony"]))
    print(f1_score(df_file["Biophony"], pred["Biophony"]))
    print(f1_score(df_file["Geophony"], pred["Geophony"]))
    # print(f1_score(df_file["Silence"], pred["Silence"]))

    print(f1_score(df_file[superclasses], pred[superclasses], average=None))
    print(f1_score(df_file[superclasses], pred[superclasses], average='macro'))







