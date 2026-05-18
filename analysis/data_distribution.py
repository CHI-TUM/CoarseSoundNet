import os
import pandas as pd
import matplotlib.pyplot as pyplot
import seaborn as sns


def get_counts(csv_path) -> dict:
    """
    Counts the occurrences of the different annotation combinations given a csv label file.
    """
    if not os.path.exists(csv_path):
        print(f"Filepath {csv_path} does not exist!")
        return None
    if not csv_path.lower().endswith(".csv"):
        print(f"File {csv_path} is not a csv!")
        return None 

    df = pd.read_csv(csv_path)
    counts = {
        "A": ((df["Anth"] == 1) & (df["Bio"] == 0) & (df["Geo"] == 0) & (df["Sil"] == 0)).sum(),
        "B": ((df["Anth"] == 0) & (df["Bio"] == 1) & (df["Geo"] == 0) & (df["Sil"] == 0)).sum(),
        "G": ((df["Anth"] == 0) & (df["Bio"] == 0) & (df["Geo"] == 1) & (df["Sil"] == 0)).sum(),
        "S": ((df["Anth"] == 0) & (df["Bio"] == 0) & (df["Geo"] == 0) & (df["Sil"] == 1)).sum(),
        "AB": ((df["Anth"] == 1) & (df["Bio"] == 1) & (df["Geo"] == 0) & (df["Sil"] == 0)).sum(),
        "AG": ((df["Anth"] == 1) & (df["Bio"] == 0) & (df["Geo"] == 1) & (df["Sil"] == 0)).sum(),
        "BG": ((df["Anth"] == 0) & (df["Bio"] == 1) & (df["Geo"] == 1) & (df["Sil"] == 0)).sum(),
        "ABG": ((df["Anth"] == 1) & (df["Bio"] == 1) & (df["Geo"] == 1) & (df["Sil"] == 0)).sum(),
        "A_t": df["Anth"].sum().astype(int),
        "B_t": df["Bio"].sum().astype(int),
        "G_t": df["Geo"].sum().astype(int),
        "S_t": df["Sil"].sum().astype(int),
        "# Samples": df.shape[0]
    }
    return counts


if __name__ == "__main__":
    datapaths = {
        "edansa": "/path/to/coarse/EDANSA-2019",
        "beambient": "/path/to/Svenja_data/segments_5s",
        "akwamo": "/path/to/akwamo_coarse_paper",
        "htsforest": "/path/to/coarse/Dominik_additional_data",
        "publicsynth": "/path/to/synthetic_data"
    }

    for data in datapaths.keys():
        print(f"\nDataset {data}:")
        path = datapaths[data]

        split_results = {}
        for split in ["train", "dev", "test"]:
            csv_path = os.path.join(path, f"{split}.csv")
            counts = get_counts(csv_path=csv_path)
            # print(counts)
            split_results[split] = counts

        df = pd.DataFrame(split_results).T
        df.loc["train+dev"] = df.loc["train"] + df.loc["dev"]
        df.loc["total"] = df.loc["train"] + df.loc["dev"] + df.loc["test"]
        print(df)
    
    print("\nBeSound original:")
    besound_path = "./analysis/processed_annotations/old_annotations.csv"
    counts = get_counts(csv_path=besound_path)
    df = pd.DataFrame([counts])
    print(df)

    print("\nBeSound version 1:")
    besound_v1_path = "./analysis/processed_annotations/adapted_th_annotations_v1.csv"
    counts = get_counts(csv_path=besound_v1_path)
    df = pd.DataFrame([counts])
    print(df)


        
        

    
