import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


if __name__ == "__main__":
    df = pd.read_csv("path/to/BESound_indices_summary_paperVersion.csv").set_index("filename")
    diversity = pd.read_csv("/path/to/Number_of_species_per_file.csv").set_index("sound.files")

    df["diversity"] = diversity["unique_annotations"]
    df = df.dropna()
    
    df["true_B"] = (
        (df["gt_new_Anth"] == 0)
        & (df["gt_new_Bio"] == 1)
        & (df["gt_new_Geo"] == 0)
    ).astype(int)
    df["true_AB"] = (
        ((df["gt_new_Anth"] == 1)
        | (df["gt_new_Bio"] == 1))
        & (df["gt_new_Geo"] == 0)
    ).astype(int)
    df["true_BG"] = (
        ((df["gt_new_Geo"] == 1)
        | (df["gt_new_Bio"] == 1))
        & (df["gt_new_Anth"] == 0)
    ).astype(int)

    df["pred_B"] = (
        (df["pred_new_Anth"] == 0)
        & (df["pred_new_Bio"] == 1)
        & (df["pred_new_Geo"] == 0)
    ).astype(int)
    df["pred_AB"] = (
        ((df["pred_new_Anth"] == 1)
        | (df["pred_new_Bio"] == 1))
        & (df["pred_new_Geo"] == 0)
    ).astype(int)
    df["pred_BG"] = (
        ((df["pred_new_Geo"] == 1)
        | (df["pred_new_Bio"] == 1))
        & (df["pred_new_Anth"] == 0)
    ).astype(int)

    plt.rcParams['text.usetex'] = True
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
    names = {
        "B": "B",
        "AB": "A \cup B",
        "BG": "B \cup G",
    }
    
    for model in ["true", "pred"]:
        for index in [
            "ADI",
            "ACI",
            "NDSI",
            "aROI",
            "nROI"
        ]:
            fig, ax = plt.subplots(1, 1)
            sns.regplot(
                data=df,
                x=index,
                y="diversity",
                ax=ax,
                fit_reg=False,
                scatter_kws={"alpha": .5, "color": "k"},
                line_kws={"color": "k"},
            )
            r = pearsonr(df[index], df["diversity"]).correlation
            sns.regplot(
                data=df,
                x=index,
                y="diversity",
                ax=ax,
                scatter=False,
                line_kws={"color": "k"},
                label=r"$x \in {}$ ($\rho={:.2f}$)".format("A \cup B \cup G", r).replace("0.", ".")
            )
            for subset in ["B", "AB", "BG"]:
                df_subset = df.loc[df[f"{model}_{subset}"] == 1]
                r = pearsonr(df_subset[index], df_subset["diversity"]).correlation
                sns.regplot(
                    data=df_subset,
                    x=index,
                    y="diversity",
                    scatter=False,
                    ax=ax,
                    scatter_kws={"alpha": .3},
                    label=r"$x \in {}$ ($\rho={:.2f}$)".format(names[subset], r).replace("0.", ".")
                )
            sns.despine(ax=ax)
            ax.set_xlabel(index, fontsize=16)
            ax.set_ylabel(r"$\alpha$-diversity", fontsize=16)
            ax.tick_params(axis="both", which="major", labelsize=14)
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(.5, 1.15),
                ncol=2,
                fontsize=12
            )
            fig.tight_layout()
            fig.savefig(f"./eco_case_study/{index}.{model}.pdf")
            fig.savefig(f"./eco_case_study/{index}.{model}.png")
