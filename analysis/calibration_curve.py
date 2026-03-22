#!/usr/bin/env python3
"""Publication-quality calibration curve analysis."""

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


def brier_score(entries_df):
    """Compute Brier Score for a set of runner entries."""
    return np.mean((entries_df["implied_prob_norm"] - entries_df["is_winner"].astype(float)) ** 2)


def expected_calibration_error(cal_df):
    """Compute Expected Calibration Error from calibration bins."""
    valid = cal_df.dropna(subset=["actual_win_rate", "mean_implied_prob"])
    if valid.empty:
        return np.nan
    total = valid["n_runners"].sum()
    ece = (valid["n_runners"] * valid["calibration_error"].abs()).sum() / total
    return ece


def main():
    """Generate calibration curve figures."""
    print("=" * 60)
    print("ANALYSIS: Calibration Curve")
    print("=" * 60)

    cal = pd.read_csv(PROCESSED_DIR + "/calibration_bins.csv")
    entries = pd.read_csv(PROCESSED_DIR + "/entries_market.csv", parse_dates=["meet_date"])
    print(f"  Calibration bins loaded: {len(cal):,} rows")
    print(f"  Entries loaded         : {len(entries):,} rows")

    # ---- Figure 1: Calibration curves for all venues + overall ---------------
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    overall = cal[cal["stratify"] == "overall"]
    ax1.plot(overall["mean_implied_prob"], overall["actual_win_rate"],
             "o-", lw=2.5, markersize=6, color="black", label="Overall", zorder=5)

    # Top venues by number of runners
    venue_counts = entries.groupby("venue").size().sort_values(ascending=False)
    top_venues = venue_counts.head(3).index.tolist()
    palette = sns.color_palette("tab10", n_colors=len(top_venues))
    for i, v in enumerate(top_venues):
        vdata = cal[cal["stratify"] == v]
        ax1.plot(vdata["mean_implied_prob"], vdata["actual_win_rate"],
                 "s--", lw=1.5, markersize=5, color=palette[i], label=v, alpha=0.8)

    ax1.set_xlabel("Mean Implied Probability")
    ax1.set_ylabel("Actual Win Rate")
    ax1.set_title("Calibration Curve: Implied Probability vs Actual Win Rate")
    ax1.legend(loc="upper left")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_aspect("equal")
    save(fig1, "calibration_curve")
    plt.close(fig1)

    # ---- Figure 2: Calibration error per bin, per venue ----------------------
    strats = ["overall"] + top_venues
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    width = 0.18
    x = np.arange(10)
    for i, s in enumerate(strats):
        sdata = cal[cal["stratify"] == s].sort_values("prob_bin_low")
        ax2.bar(x + i * width, sdata["calibration_error"].values, width,
                label=s, alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x + width * (len(strats) - 1) / 2)
    labels = [f"{lo:.1f}–{hi:.1f}" for lo, hi in
              zip(overall["prob_bin_low"], overall["prob_bin_high"])]
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_xlabel("Implied Probability Bin")
    ax2.set_ylabel("Calibration Error (actual − implied)")
    ax2.set_title("Calibration Error by Probability Bin")
    ax2.legend()
    save(fig2, "calibration_error_bars")
    plt.close(fig2)

    # ---- Compute Brier Scores and ECE ----------------------------------------
    print()
    print("Venue Efficiency Metrics")
    print("-" * 60)
    print(f"  {'Venue':>15s}  {'Brier Score':>12s}  {'ECE':>10s}  {'n_runners':>10s}")
    print("  " + "-" * 50)

    results = []
    # Overall
    bs_all = brier_score(entries)
    ece_all = expected_calibration_error(cal[cal["stratify"] == "overall"])
    results.append(("Overall", bs_all, ece_all, len(entries)))

    for v in sorted(entries["venue"].dropna().unique()):
        ve = entries[entries["venue"] == v]
        vc = cal[cal["stratify"] == v]
        bs = brier_score(ve)
        ece = expected_calibration_error(vc)
        results.append((v, bs, ece, len(ve)))

    results.sort(key=lambda x: x[1])
    for name, bs, ece, n in results:
        print(f"  {name:>15s}  {bs:12.6f}  {ece:10.6f}  {n:>10,}")

    print()
    most_efficient = results[0][0] if results[0][0] != "Overall" else results[1][0]
    least_efficient = results[-1][0]
    print(f"  Most efficient venue : {most_efficient}")
    print(f"  Least efficient venue: {least_efficient}")


if __name__ == "__main__":
    main()
