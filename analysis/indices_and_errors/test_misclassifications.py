import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("/path/to/BESound_indices_summary_paperVersion.csv").set_index("filename")

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


    for type in ["fp", "fn"]:
        for target in [
            "Anth",
            "Bio",
            "Geo"
        ]:
            fig, ax = plt.subplots(1, 1)
            if type == "fp":
                df[target] = ~df[f"gt_new_{target}"] & df[f"pred_new_{target}"]
            else:
                df[target] = df[f"gt_new_{target}"] & ~df[f"pred_new_{target}"]
            g = sns.barplot(
                data=df,
                x="class",
                y=target,
                ax=ax,
                order=hue_order,
                palette=colors,
                errorbar=None,
                native_scale=True
            )
            sns.despine(ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel(type.upper(), fontsize=16)
            ax.set_title(target, fontsize=18, pad=8)
            ax.tick_params(axis="both", which="major", labelsize=14)
            plt.tight_layout()
            plt.savefig(f"./misclassifications/{target}.{type}.svg", transparent=True)