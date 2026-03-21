#!/usr/bin/env python3
"""Build calibration bins from entries_market.csv for calibration curve analysis."""

import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = "../data/processed"

BIN_EDGES = np.arange(0.0, 1.1, 0.1)  # 0.0, 0.1, ..., 1.0 → 10 bins


def compute_calibration(df_subset, stratify_label):
    """Compute calibration bins for a subset of runners."""
    rows = []
    for i in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
        mask = (df_subset["implied_prob_norm"] >= lo) & (df_subset["implied_prob_norm"] < hi)
        sub = df_subset.loc[mask]
        n_runners = len(sub)
        n_winners = int(sub["is_winner"].sum())
        actual_win_rate = n_winners / n_runners if n_runners > 0 else np.nan
        mean_implied = sub["implied_prob_norm"].mean() if n_runners > 0 else np.nan
        cal_error = (actual_win_rate - mean_implied) if n_runners > 0 else np.nan
        rows.append({
            "stratify": stratify_label,
            "prob_bin_low": lo,
            "prob_bin_high": hi,
            "prob_bin_mid": (lo + hi) / 2,
            "n_runners": n_runners,
            "n_winners": n_winners,
            "actual_win_rate": actual_win_rate,
            "mean_implied_prob": mean_implied,
            "calibration_error": cal_error,
        })
    return pd.DataFrame(rows)


def main():
    """Build calibration bins."""
    print("=" * 60)
    print("PIPELINE STEP 4: Build Calibration Bins")
    print("=" * 60)

    entries_path = PROCESSED_DIR + "/entries_market.csv"

    df = pd.read_csv(entries_path, parse_dates=["meet_date"])
    print(f"  Loaded {len(df):,} runner entries")

    # --- Overall calibration -------------------------------------------------
    parts = [compute_calibration(df, "overall")]

    # --- Per-venue calibration -----------------------------------------------
    venues = sorted(df["venue"].dropna().unique())
    for v in venues:
        vdf = df[df["venue"] == v]
        parts.append(compute_calibration(vdf, v))

    out = pd.concat(parts, ignore_index=True)

    # --- Summary -------------------------------------------------------------
    print()
    print("Calibration Summary (Overall)")
    print("-" * 60)
    overall = out[out["stratify"] == "overall"]
    for _, row in overall.iterrows():
        print(
            f"  [{row['prob_bin_low']:.1f}, {row['prob_bin_high']:.1f})  "
            f"n={row['n_runners']:>6,}  "
            f"win_rate={row['actual_win_rate']:.4f}  "
            f"implied={row['mean_implied_prob']:.4f}  "
            f"error={row['calibration_error']:+.4f}"
        )

    # --- Write ---------------------------------------------------------------
    out_path = PROCESSED_DIR + "/calibration_bins.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWritten → {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
