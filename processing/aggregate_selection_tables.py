import os
import pandas as pd
import argparse
from glob import glob
from tqdm import tqdm


def postprocess_annotations(df: pd.DataFrame) -> pd.DataFrame:

    df["Annotation"] = df["Annotation"].str.strip()
    df.rename(columns={"filename": "file"}, inplace=True)
    df = df.set_index(["file", "Selection"])
    df = df.loc[~df.index.duplicated()]
    df = df.reset_index()

    # df["Plot"] = df["file"].apply(lambda x: x.split("_")[1])
    # df["Date"] = df["file"].apply(lambda x: x.split("_")[0].split("-")[1])
    df["Plot"] = df["file"].apply(lambda x: x.split("/")[-1].split("_")[1])
    df["Date"] = df["file"].apply(lambda x: x.split("/")[-1].split("_")[0].split("-")[1])


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
        "Silence",
        "Background",
        "Backround"
    ]
    exclude = [
        "Other",
        "Interference",
    ]

    df["Duration"] = df["End Time (s)"] - df["Begin Time (s)"]
    df = df.loc[~df["Annotation"].isna()]

    silent_files = df.loc[df["Annotation"] == "Silence", "file"].values

    df = df.loc[~df["Annotation"].isin(exclude)]
    df["Superclass"] = "Biophony"
    df.loc[df["Annotation"].isin(geophony), "Superclass"] = "Geophony"
    df.loc[df["Annotation"].isin(anthropophony), "Superclass"] = "Anthropophony"
    df.loc[df["Annotation"].isin(silence), "Superclass"] = "Silence"
    df["Hours"] = df["Duration"] / 3600

    print(df.head())
    return df


if __name__=='__main__':
    parser = argparse.ArgumentParser("Aggregate selection tables.")
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="The directory of which the selection tables shall be aggregated."
    )
    args = parser.parse_args()
    prediction_dir = args.dir 

    if not os.path.exists(prediction_dir):
        print(f"The specified directory does not exist: {prediction_dir}.")
        print("Abort.")
        exit()

    if os.path.exists(os.path.join(prediction_dir, "aggreagted_tables.csv")):
        print(f"results.csv already exists in {prediction_dir}.")
    else:
        files = glob(os.path.join(prediction_dir, "**/*.selections.txt"), recursive=True)
        df = pd.DataFrame()

        for file in tqdm(files, total=len(files), desc="Read"):
            local_df = pd.read_csv(file, sep="\t")
            relative_path = os.path.relpath(file, prediction_dir)
            # local_df["filename"] = os.path.basename(file).split(".")[0]
            local_df["filename"] = relative_path.split(".")[0] + ".wav"
            df = pd.concat((df, local_df), ignore_index=True)

        # Postprocess the concatenated dataframe
        df = postprocess_annotations(df=df)
        print(df["Annotation"].unique())
        
        df.to_csv(os.path.join(prediction_dir, "aggregated_tables.csv"), index=False)
        print("Created results.csv.")

        # Caution: Silence annotations might be active even though other classes are annotated! Use scripts/adjust_silence_annotations.py