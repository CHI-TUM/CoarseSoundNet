import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr


if __name__ == "__main__":
    df = pd.read_csv(
        "/path/to/BESound_indices_summary_paperVersion.csv"
    ).set_index("filename")
    diversity = pd.read_csv(
        "/path/to/Number_of_species_per_file.csv"
    ).set_index("sound.files")

    df["diversity"] = diversity["unique_annotations"]
    df = df.dropna()

    df["true_B"] = (
        (df["gt_new_Anth"] == 0) & (df["gt_new_Bio"] == 1) & (df["gt_new_Geo"] == 0)
    ).astype(int)
    df["true_AB"] = (
        ((df["gt_new_Anth"] == 1) | (df["gt_new_Bio"] == 1)) & (df["gt_new_Geo"] == 0)
    ).astype(int)
    df["true_BG"] = (
        ((df["gt_new_Geo"] == 1) | (df["gt_new_Bio"] == 1)) & (df["gt_new_Anth"] == 0)
    ).astype(int)

    df["pred_B"] = (
        (df["pred_new_Anth"] == 0) & (df["pred_new_Bio"] == 1) & (df["pred_new_Geo"] == 0)
    ).astype(int)
    df["pred_AB"] = (
        ((df["pred_new_Anth"] == 1) | (df["pred_new_Bio"] == 1)) & (df["pred_new_Geo"] == 0)
    ).astype(int)
    df["pred_BG"] = (
        ((df["pred_new_Geo"] == 1) | (df["pred_new_Bio"] == 1)) & (df["pred_new_Anth"] == 0)
    ).astype(int)

    plt.rcParams["text.usetex"] = True

    indices = ["ACI", "ADI", "NDSI"]
    models = ["true", "pred"]

    # variant -> (column suffix or None for "all", label, color)
    variants = [
        (None, r"$A \cup B \cup G$", "black"),
        ("B", r"$B$", "green"),
        ("AB", r"$A \cup B$", "darkcyan"),
        ("BG", r"$B \cup G$", "saddlebrown"),
    ]

    def corr_stats(x, y):
        """n, Pearson r, Spearman rho for two aligned series (NaNs dropped)."""
        mask = x.notna() & y.notna()
        x, y = x[mask], y[mask]
        n = len(x)
        if n < 3:
            return n, np.nan, np.nan
        r_p = pearsonr(x, y).statistic
        r_s = spearmanr(x, y).statistic
        return n, r_p, r_s

    def subset_for(model, suffix):
        return df if suffix is None else df.loc[df[f"{model}_{suffix}"] == 1]

    # ---------------------------------------------------------------
    # Figure 1: 2x3 regression grid
    # ---------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 3, figsize=(14, 8), sharex="col", sharey=True)

    for row, model in enumerate(models):
        for col, index in enumerate(indices):
            ax = axes1[row, col]

            # scatter of the full ("all") data as background
            sns.regplot(
                data=df,
                x=index,
                y="diversity",
                ax=ax,
                fit_reg=False,
                scatter_kws={"alpha": 0.4, "color": "k", "s": 15},
            )

            for suffix, label, color in variants:
                sub = subset_for(model, suffix)
                sns.regplot(
                    data=sub,
                    x=index,
                    y="diversity",
                    ax=ax,
                    scatter=False,
                    line_kws={"color": color},
                    label=label,
                )

            sns.despine(ax=ax)
            ax.tick_params(axis="both", which="major", labelsize=13)
            if row == 0:
                ax.set_title(index, fontsize=17)
            ax.set_xlabel(index if row == 1 else "", fontsize=15)
            ax.set_ylabel(r"$\alpha$-diversity" if col == 0 else "", fontsize=15)

    handles, labels = axes1[0, 0].get_legend_handles_labels()
    fig1.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=4, fontsize=15, frameon=False,
    )
    fig1.tight_layout(rect=[0.03, 0, 1, 0.94])
    fig1.savefig("./eco_case_study_adapted/regression_grid.pdf", bbox_inches="tight")
    fig1.savefig("./eco_case_study_adapted/regression_grid.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # ---------------------------------------------------------------
    # Figure 2: 2x3 dumbbell grid, Pearson vs Spearman
    # ---------------------------------------------------------------
    variant_colors = {suffix: color for suffix, _, color in variants}
    cat_suffixes = [suffix for suffix, _, _ in variants]
    cat_labels = [label for _, label, _ in variants]
    y_pos = list(range(len(variants) - 1, -1, -1))  # "all" at top, matching the sketch

    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 6.5), sharex=True)

    summary_rows = []

    for row, model in enumerate(models):
        # sample size per subset depends only on model, not on the acoustic index
        n_by_suffix = {suffix: len(subset_for(model, suffix)) for suffix in cat_suffixes}
        row_yticklabels = [
            f"{label} ($n={n_by_suffix[suffix]}$)"
            for suffix, label in zip(cat_suffixes, cat_labels)
        ]

        for col, index in enumerate(indices):
            ax = axes2[row, col]

            for y, suffix, label in zip(y_pos, cat_suffixes, cat_labels):
                color = variant_colors[suffix]
                sub = subset_for(model, suffix)
                n, r_p, r_s = corr_stats(sub[index], sub["diversity"])
                summary_rows.append(
                    {
                        "model": model, "index": index,
                        "subset": "all" if suffix is None else suffix,
                        "n": n, "pearson_r": r_p, "spearman_rho": r_s,
                    }
                )
                ax.plot([r_p, r_s], [y, y], color=color, lw=1, alpha=0.6, zorder=1)
                ax.scatter(r_p, y, marker="o", facecolor=color, edgecolor=color, s=55, zorder=3)
                ax.scatter(r_s, y, marker="o", facecolor="white", edgecolor=color, s=55, zorder=3, linewidths=1.8)
                ax.text(r_p, y + 0.22, f"{r_p:.2f}", ha="center", va="bottom", fontsize=10, color=color)
                ax.text(r_s, y - 0.22, f"{r_s:.2f}", ha="center", va="top", fontsize=10, color=color)

            ax.axvline(0, color="gray", lw=0.6, ls="--", zorder=0)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-0.6, len(variants) - 0.4)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(row_yticklabels if col == 0 else [], fontsize=14)
            sns.despine(ax=ax, left=(col != 0))
            ax.tick_params(axis="both", which="major", labelsize=13, labelleft=(col == 0))
            if row == 0:
                ax.set_title(index, fontsize=17)
            if row == 1:
                ax.set_xlabel("correlation", fontsize=14)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markeredgecolor="black", markersize=9, label="Pearson"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=9, label="Spearman"),
    ]
    fig2.legend(
        handles=legend_handles,
        loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=2, fontsize=15, frameon=False,
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    fig2.savefig("./eco_case_study_adapted/correlation_dumbbell.pdf", bbox_inches="tight")
    fig2.savefig("./eco_case_study_adapted/correlation_dumbbell.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # Summary table for the rebuttal / supplementary material
    pd.DataFrame(summary_rows).to_csv(
        "./eco_case_study_adapted/correlation_summary.csv", index=False
    )