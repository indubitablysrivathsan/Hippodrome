#!/usr/bin/env python3
"""Publication-quality figures for entropy distribution analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = "../data/processed"
FIG_DIR = "../outputs/figures"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

SAVEKW = dict(dpi=150, bbox_inches="tight")


def save(fig, name):
    """Save a figure as PNG and PDF."""
    fig.savefig(FIG_DIR + f"/{name}.png", **SAVEKW)
    fig.savefig(FIG_DIR + f"/{name}.pdf", bbox_inches="tight")
    print(f"  Saved {name}.png / .pdf")


def main():
    """Generate entropy distribution figures."""
    print("=" * 60)
    print("ANALYSIS: Entropy Distribution")
    print("=" * 60)

    df = pd.read_csv(PROCESSED_DIR + "/race_features.csv", parse_dates=["meet_date"])
    print(f"  Loaded {len(df):,} races")

    # ---- Figure 1: Distribution of entropy_H --------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.histplot(df["entropy_H"].dropna(), bins=60, kde=True, ax=ax1, color="steelblue")
    mu, sigma = df["entropy_H"].mean(), df["entropy_H"].std()
    ax1.axvline(mu, color="red", ls="--", lw=1.5, label=f"Mean = {mu:.3f}")
    ax1.axvline(mu - sigma, color="orange", ls=":", lw=1, label=f"−1σ = {mu - sigma:.3f}")
    ax1.axvline(mu + sigma, color="orange", ls=":", lw=1, label=f"+1σ = {mu + sigma:.3f}")
    ax1.set_xlabel("Shannon Entropy H (bits)")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Race Entropy")
    ax1.legend()
    save(fig1, "entropy_distribution_hist")
    plt.close(fig1)

    # ---- Figure 2: Entropy by venue -----------------------------------------
    venue_order = df.groupby("venue")["entropy_H"].median().sort_values().index
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="venue", y="entropy_H", order=venue_order, ax=ax2,
                palette="viridis")
    ax2.set_xlabel("Venue")
    ax2.set_ylabel("Shannon Entropy H (bits)")
    ax2.set_title("Entropy Distribution by Venue")
    ax2.tick_params(axis="x", rotation=45)
    save(fig2, "entropy_by_venue_box")
    plt.close(fig2)

    # ---- Figure 3: Entropy by track condition --------------------------------
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    cats = ["Good", "Soft", "Heavy", "Unknown"]
    present = [c for c in cats if c in df["track_condition_category"].values]
    sns.violinplot(data=df[df["track_condition_category"].isin(present)],
                   x="track_condition_category", y="entropy_H", order=present,
                   palette="muted", ax=ax3)
    ax3.set_xlabel("Track Condition Category")
    ax3.set_ylabel("Shannon Entropy H (bits)")
    ax3.set_title("Entropy by Track Condition")
    save(fig3, "entropy_by_track_condition")
    plt.close(fig3)

    # ---- Figure 4: Entropy vs field size ------------------------------------
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    venues = df["venue"].unique()
    palette = sns.color_palette("tab10", n_colors=len(venues))
    for i, v in enumerate(sorted(venues)):
        vdf = df[df["venue"] == v]
        ax4.scatter(vdf["field_size"], vdf["entropy_H"], s=8, alpha=0.4,
                    label=v, color=palette[i])
    # Overall regression
    valid = df.dropna(subset=["field_size", "entropy_H"])
    slope, intercept, r, p, se = stats.linregress(valid["field_size"], valid["entropy_H"])
    x_reg = np.linspace(valid["field_size"].min(), valid["field_size"].max(), 100)
    ax4.plot(x_reg, slope * x_reg + intercept, "k--", lw=2,
             label=f"OLS: r={r:.3f}, p={p:.2e}")
    ax4.set_xlabel("Field Size (runners)")
    ax4.set_ylabel("Shannon Entropy H (bits)")
    ax4.set_title("Entropy vs Field Size")
    ax4.legend(fontsize=7, ncol=2)
    save(fig4, "entropy_vs_field_size")
    plt.close(fig4)

    # ---- Summary stats ------------------------------------------------------
    print()
    print("Summary Statistics by Venue")
    print("-" * 60)
    summary = (
        df.groupby("venue")["entropy_H"]
        .agg(["count", "mean", "std", "min", "max"])
        .sort_values("mean", ascending=False)
    )
    print(summary.to_string())
    print()


if __name__ == "__main__":
    main()
