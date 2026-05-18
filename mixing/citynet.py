import argparse
import glob
import os
import pandas as pd

category_map = {
    # Biophony → "Bio"
    "bird": "Bio",
    "invertebrate": "Bio",
    "animal": "Bio",
    "barking dog": "Bio",
    "dog barking": "Bio",
    "amphibian": "Bio",
    "dog bark": "Bio",

    # Geophony → "Geo"
    "wind": "Geo",
    "rain": "Geo",
    "vegetation": "Geo",
    "rainfall on vegetation": "Geo",

    # Anthropophony → "Anth"
    "road traffic": "Anth",
    "airplane": "Anth",
    "rail traffic": "Anth",
    "Mix traffic": "Anth"
}

# biophony = ["bird", "invertebrate", "animal", "barking dog", "dog barking", "amphibian", "dog bark"]
# geophony = ["wind", "rain", "vegetation", "rainfall on vegetation"]
# anthropohpony = ["road traffic", "airplane", "rail traffic", "Mix traffic"] 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    files = glob.glob(
        os.path.join(
            args.src,
            "**",
            "*.csv"
        ),
        recursive=True
    )
    print(files[-1])
    df = pd.DataFrame()
    for file in files:
        local_df = pd.read_csv(file)
        local_df = local_df.rename(
            columns={
                "Filename": "file",
                "Label": "meta",
                "LabelStartTime_Seconds": "start",
                "LabelEndTime_Seconds": "end"
            },
        )
        local_df["split"] = "train" if "train" in file.lower() else "test"
        # print(local_df.head())
        df = pd.concat((df, local_df), ignore_index=True)
    
    print("Number of test files before filtering: ", len(df[df["split"] == "test"]))
    df = df[["file", "start", "end", "meta", "split"]]
    df = df[df["meta"].isin(category_map.keys())]
    print("Number of test files after filtering: ", len(df[df["split"] == "test"]))
    df["supercategory"] = df["meta"].map(category_map)
    df = df.reset_index(drop=True)
    print("Unique classes: ", df["meta"].unique())
    print(df.head(), df.shape)
    df.to_csv(os.path.join(args.dst, "citynet.csv"), index=False)

    


    