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
    root: str = "/path/to/FromBEforests_allAnnotatedFiles"
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

        # df = df.groupby('file')['Annotation'].apply(lambda x: x.unique()).reset_index()
        # df = df.explode('Annotation')
        # df.rename(columns={'file': 'filename'}, inplace=True)
        # df["filename"] += ".wav"
        # print(df.head(15))
        # df.to_csv(fp, index=False)

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

        # def map_classes(x):
        #     anth = int(np.sum(x["Superclass"] == "Anthropophony") > 0)
        #     bio = int(np.sum(x["Superclass"] == "Biophony") > 0)
        #     geo = int(np.sum(x["Superclass"] == "Geophony") > 0)
        #     sil = int((anth + bio + geo == 0))
        #     return pd.Series({
        #         "Anthropophony": anth,
        #         "Biophony": bio,
        #         "Geophony": geo,
        #         "Silence": sil,
        #     })
        # df = df.loc[df["Annotation"].isin(["Anthropohony", "Anthropophony", "Geophony", "Geophpny", "Geophpony", "Biophony", "Silence"])]
        print(df.head())
        df.to_csv(fp, index=False)

    return df


if __name__=='__main__':
    predictions_path = "/path/to/predictions"
    variant = "max"
    version = "adaptedAnnotations"
    model = "hf"


    if model != "BE":
        df = load_annotations(fp="./annotations_BEForest.csv")
        df["file"] += ".wav"
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
        df_file = df.groupby("file").apply(map_classes).reset_index()
        df_file = df_file.set_index("file")
    else:
        # for testing BE model on akwamo
        df_file = pd.read_csv("/path/to/BE_data/test.csv")
        df_file = df_file.rename(columns={"filename": "file"})
        df_file.set_index("file", inplace=True)
        df_file = df_file.rename(columns={
            "Anth": "Anthropophony",
            "Bio": "Biophony",
            "Geo": "Geophony",
            "Sil": "Silence",
        })

        df_file = df_file.replace(2, 1)
        df_file["Silence"] = df_file.apply(lambda x: int((x["Silence"] & x.sum() == 1) | (x.sum() == 0)), axis=1)
    
    if version == "adaptedAnnotations":
        df_adapted_thresholds_v1 = pd.read_csv("./analysis/processed_annotations/adapted_th_annotations_v1.csv").set_index("filename")
        df_file = df_adapted_thresholds_v1
        df_file = df_file.rename(columns={
            "Anth": "Anthropophony",
            "Bio": "Biophony",
            "Geo": "Geophony",
            "Sil": "Silence",
        })
    print(df_file.head(), df_file.shape)

    # Load the predictions
    pred = pd.read_csv(os.path.join(predictions_path, "results.csv"))
    # pred.rename(columns={"filename": "file"}, inplace=True)
    pred = pred.rename(columns={
        "Anth": "Anthropophony",
        "Bio": "Biophony",
        "Geo": "Geophony",
        "Sil": "Silence",
    })
    superclasses = [
        "Anthropophony",
        "Biophony",
        "Geophony",
        "Silence",
    ]
    pred = pred.drop(columns=["prediction", "output"])
    

    pred.drop(columns=["offset"], inplace=True)
    # pred = pred.groupby("file").max()
    pred = pred.groupby("filename").max()
    
    # Align indices
    common_index = df_file.index.intersection(pred.index)
    df_file = df_file.loc[common_index]
    df_file = df_file.sort_index()
    pred = pred.loc[common_index]
    pred = pred.sort_index()
    print(pred.head(), pred.shape)

    colors = {
        "A": "cornflowerblue",
        "B": "forestgreen",
        "G": "maroon",
        "S": "gray",
        "A+B": "aqua",
        "A+G": "midnightblue",
        "B+G": "indigo",
        "A+B+G": "black",
    }
    print(df_file)
    width = .35
    bar_df = df_file.copy()
    bar_df["A"] = ((bar_df["Anthropophony"] == 1) & (bar_df["Biophony"] == 0) & (bar_df["Geophony"] == 0)).astype(int)
    bar_df["B"] = ((bar_df["Anthropophony"] == 0) & (bar_df["Biophony"] == 1) & (bar_df["Geophony"] == 0)).astype(int)
    bar_df["G"] = ((bar_df["Anthropophony"] == 0) & (bar_df["Biophony"] == 0) & (bar_df["Geophony"] == 1)).astype(int)
    bar_df["S"] = ((bar_df["Anthropophony"] == 0) & (bar_df["Biophony"] == 0) & (bar_df["Geophony"] == 0)).astype(int)
    bar_df["A+B"] = ((bar_df["Anthropophony"] == 1) & (bar_df["Biophony"] == 1) & (bar_df["Geophony"] == 0)).astype(int)
    bar_df["A+G"] = ((bar_df["Anthropophony"] == 1) & (bar_df["Biophony"] == 0) & (bar_df["Geophony"] == 1)).astype(int)
    bar_df["B+G"] = ((bar_df["Anthropophony"] == 0) & (bar_df["Biophony"] == 1) & (bar_df["Geophony"] == 1)).astype(int)
    bar_df["A+B+G"] = ((bar_df["Anthropophony"] == 1) & (bar_df["Biophony"] == 1) & (bar_df["Geophony"] == 1)).astype(int)
    # bar_df["I"] = ((bar_df["Anthropophony"] == 1) & (bar_df["Insect"] == 1) & (bar_df["Geophony"] == 1)).astype(int)
    labels = ["A", "B", "G", "S", "A+B", "A+G", "B+G", "A+B+G"]
    
    # targets = pred.columns  # superclasses
    targets = superclasses
    fig, axes = plt.subplots(len(targets), 1, figsize=[3 * len(targets), len(targets)])
    for ax_index, class_name in enumerate(targets):
        bar_df["confidence"] = pred[class_name]
        ax = axes[ax_index]
        for index, label in enumerate(labels):
            
            ax.bar(
                [index - width/2, index + width/2],
                [bar_df.loc[bar_df[label] == 0, "confidence"].mean(), bar_df.loc[bar_df[label] == 1, "confidence"].mean()],
                color=colors[label],
                width=width
            )
            s = f'{bar_df.loc[bar_df[label] == 0, "confidence"].mean():.2f}'[1:]
            ax.text(
                index - width,
                bar_df.loc[bar_df[label] == 0, "confidence"].mean() + 0.01,
                s
            )

            s = f'{bar_df.loc[bar_df[label] == 1, "confidence"].mean():.2f}'[1:]
            ax.text(
                index,
                bar_df.loc[bar_df[label] == 1, "confidence"].mean() + 0.01,
                s
            )
        ax.set_xticks(range(len(labels)), labels=labels)
        ax.set_ylabel("Confidence")
        ax.set_title(f"Max {class_name} confidence")
        sns.despine(ax=ax)
    plt.tight_layout()
    # plt.savefig(os.path.join(".", "edansa_only_ft_iNat32k.barplot.png"))
    plt.savefig(os.path.join("./analysis", "hf_ws_10s_hs_1s.png"))
    plt.close()
    pred.reset_index(inplace=True)
    exit()





    # individual thresholds
    thresholds = {
        "Anthropophony": 0.5,
        "Biophony": 0.3,
        "Geophony": 0.3,
        "Silence": 0.3
    }
    
    # if max
    if variant == "max":
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

    
    


    pred["Silence"] = pred.apply(lambda x: int((x["Silence"] & x.sum() == 1) | (x.sum() == 0)), axis=1)
    print(pred.head(), pred.shape)
    # print(df_file, pred)
    # pred = pred.drop(columns=["Silence"])
    # df_file = df_file.drop(columns=["Silence"])
    print(df_file[superclasses].sum())


    print(f1_score(df_file["Anthropophony"], pred["Anthropophony"]))
    print(f1_score(df_file["Biophony"], pred["Biophony"]))
    print(f1_score(df_file["Geophony"], pred["Geophony"]))
    print(f1_score(df_file["Silence"], pred["Silence"]))

    print(f1_score(df_file[superclasses], pred[superclasses], average=None))
    print(f1_score(df_file[superclasses], pred[superclasses], average='macro'))







