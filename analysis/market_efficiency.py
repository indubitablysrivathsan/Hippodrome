#!/usr/bin/env python3
"""Publication-quality figures for market efficiency analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths & style
# ---------------------------------------------------------------------------
PROCESSED_DIR = "../data/processed"
FIG_DIR = "../outputs/figures"

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
    """Generate market efficiency figures."""
    print("=" * 60)
    print("ANALYSIS: Market Efficiency")
    print("=" * 60)

    df = pd.read_csv(PROCESSED_DIR + "/race_features.csv", parse_dates=["meet_date"])
    print(f"  Loaded {len(df):,} races")

    venues_sorted = sorted(df["venue"].dropna().unique())

    # ---- Figure 1: Overround by venue ----------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="venue", y="overround",
                order=venues_sorted, palette="Set2", ax=ax1)
    ax1.set_xlabel("Venue")
    ax1.set_ylabel("Overround (bookmaker margin)")
    ax1.set_title("Overround Distribution by Venue")
    ax1.tick_params(axis="x", rotation=45)
    save(fig1, "overround_by_venue")
    plt.close(fig1)

    # ---- Figure 2: Overround vs Entropy --------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("tab10", n_colors=len(venues_sorted))
    for i, v in enumerate(venues_sorted):
        vdf = df[df["venue"] == v]
        ax2.scatter(vdf["entropy_H"], vdf["overround"], s=8, alpha=0.35,
                    label=v, color=palette[i])
    ax2.set_xlabel("Shannon Entropy H (bits)")
    ax2.set_ylabel("Overround")
    ax2.set_title("Overround vs Entropy")
    ax2.legend(fontsize=7, ncol=2)
    save(fig2, "overround_vs_entropy")
    plt.close(fig2)

    # ---- Figure 3: Favourite win rate by venue (with 95% CI) -----------------
    venue_fav = (
        df.groupby("venue")
        .agg(n=("winner_was_favourite", "count"),
             fav_wins=("winner_was_favourite", "sum"))
        .reset_index()
    )
    venue_fav["rate"] = venue_fav["fav_wins"] / venue_fav["n"]
    venue_fav["ci95"] = 1.96 * np.sqrt(
        venue_fav["rate"] * (1 - venue_fav["rate"]) / venue_fav["n"]
    )
    venue_fav = venue_fav.sort_values("rate", ascending=False)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(venue_fav))
    ax3.bar(x, venue_fav["rate"], yerr=venue_fav["ci95"],
            capsize=4, color="steelblue", edgecolor="white")
    ax3.set_xticks(x)
    ax3.set_xticklabels(venue_fav["venue"], rotation=45, ha="right")
    ax3.set_ylabel("Favourite Win Rate")
    ax3.set_title("Favourite Win Rate by Venue (95% CI)")
    ax3.axhline(venue_fav["rate"].mean(), ls="--", color="red", lw=1,
                label=f"Mean = {venue_fav['rate'].mean():.2%}")
    ax3.legend()
    save(fig3, "favourite_win_rate_by_venue")
    plt.close(fig3)

    # ---- Figure 4: Upset magnitude distribution ------------------------------
    fig4, ax4 = plt.subplots(figsize=(9, 5))
    winner_ranks = df["winner_prob_rank"].dropna().astype(int)
    rank_counts = winner_ranks.value_counts().sort_index()
    ax4.bar(rank_counts.index, rank_counts.values, color="coral", edgecolor="white")
    ax4.set_xlabel("Winner's Probability Rank (1 = favourite)")
    ax4.set_ylabel("Number of Races")
    ax4.set_title("Distribution of Winner Probability Rank")
    ax4.set_xlim(0.5, min(rank_counts.index.max(), 20) + 0.5)
    save(fig4, "winner_prob_rank_dist")
    plt.close(fig4)

    # ---- Print summary -------------------------------------------------------
    print()
    print("Overround Summary by Venue")
    print("-" * 50)
    print(df.groupby("venue")["overround"].agg(["mean", "std"]).to_string())
    print()
    print("Favourite Win Rates")
    print("-" * 50)
    for _, row in venue_fav.iterrows():
        print(f"  {row['venue']:>12s}: {row['rate']:.2%}  (n={row['n']:,})")
    print()
    underdog = df[df["winner_prob_rank"] >= 4]
    print(f"Underdog wins (rank ≥ 4): {len(underdog):,} / {len(df):,} "
          f"= {len(underdog) / len(df):.2%}")


if __name__ == "__main__":
    main()
