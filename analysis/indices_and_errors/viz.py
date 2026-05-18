import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("/path/to/BESound_indices_summary_paperVersion.csv")

    df["class"] = None
    df.loc[
        (df["gt_new_Anth"] == 1)
        & (df["gt_new_Bio"] == 0)
        & (df["gt_new_Geo"] == 0),
        "class"
    ] = "A"
    df.loc[
        (df["gt_new_Anth"] == 0)
        & (df["gt_new_Bio"] == 1)
        & (df["gt_new_Geo"] == 0),
        "class"
    ] = "B"
    df.loc[
        (df["gt_new_Anth"] == 0)
        & (df["gt_new_Bio"] == 0)
        & (df["gt_new_Geo"] == 1),
        "class"
    ] = "G"
    df.loc[
        (df["gt_new_Anth"] == 0)
        & (df["gt_new_Bio"] == 0)
        & (df["gt_new_Geo"] == 0),
        "class"
    ] = "S"
    df.loc[
        (df["gt_new_Anth"] == 0)
        & (df["Insect_duration"] > 5)
        & (df["gt_new_Geo"] == 0),
        "class"
    ] = "I"
    df.loc[
        (df["gt_new_Anth"] == 1)
        & (df["gt_new_Bio"] == 1)
        & (df["gt_new_Geo"] == 0),
        "class"
    ] = "AB"
    df.loc[
        (df["gt_new_Anth"] == 1)
        & (df["gt_new_Bio"] == 0)
        & (df["gt_new_Geo"] == 1),
        "class"
    ] = "AG"

    df.loc[
        (df["gt_new_Anth"] == 0)
        & (df["gt_new_Bio"] == 1)
        & (df["gt_new_Geo"] == 1),
        "class"
    ] = "BG"
    df.loc[
        (df["gt_new_Anth"] == 1)
        & (df["gt_new_Bio"] == 1)
        & (df["gt_new_Geo"] == 1),
        "class"
    ] = "ABG"

    df = df.rename(
        columns={
            "Anth_max_prob": "Anth",
            "Bio_max_prob": "Bio",
            "Geo_max_prob": "Geo"
        }
    )
    hue_order = ["A", "B", "G", "S", "I", "AB", "AG", "BG", "ABG"]
    colors = {
        "A": "royalblue",
        "B": "green",
        "G": "crimson",
        "S": "gray",
        "I": "limegreen",
        "AB": "darkcyan",
        "AG": "darkviolet",
        "BG": "saddlebrown",
        "ABG": "maroon"
    }

    for target in [
        "ACI",
        "ADI",
        "NDSI",
        "Anth",
        "Bio",
        "Geo",
        "aROI",
        "nROI"
    ]:
        fig, ax = plt.subplots(1, 1)
        g = sns.boxplot(
            data=df,
            x="class",
            y=target,
            ax=ax,
            order=hue_order,
            notch=True,
            showfliers=False,
            palette=colors
            # outliers=False
            # ci=None
        )
        # g.bar_label(g.containers[0])
        sns.despine(ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel(target, fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=14)
        plt.tight_layout()
        plt.savefig(f"./indices/{target}_paper.pdf")
        plt.savefig(f"./indices/{target}_paper.png")