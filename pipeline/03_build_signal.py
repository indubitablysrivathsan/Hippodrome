#!/usr/bin/env python3
"""Build the chronological entropy signal from race features."""

import sys

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = "../data/processed"


def main():
    """Build entropy signal time series."""
    print("=" * 60)
    print("PIPELINE STEP 3: Build Entropy Signal")
    print("=" * 60)

    feat_path = PROCESSED_DIR + "/race_features.csv"

    df = pd.read_csv(feat_path, parse_dates=["meet_date"])
    print(f"  Loaded {len(df):,} races")

    # --- Sort strictly by date then race number ------------------------------
    df = df.sort_values(["meet_date", "race_no"]).reset_index(drop=True)

    # --- Race sequence id (1-indexed) ----------------------------------------
    df["race_seq_id"] = range(1, len(df) + 1)

    # --- Entropy bins (tertiles) ---------------------------------------------
    df["entropy_bin"] = pd.qcut(
        df["entropy_H_norm"],
        q=3,
        labels=["low", "medium", "high"],
    )

    # --- Venue code ----------------------------------------------------------
    le = LabelEncoder()
    df["venue_code"] = le.fit_transform(df["venue"].fillna("Unknown"))

    # --- Select output columns -----------------------------------------------
    out = df[[
        "race_seq_id", "meet_date", "venue", "race_no",
        "field_size", "entropy_H", "entropy_H_norm", "entropy_bin",
        "overround", "winner_prob", "winner_was_favourite",
        "season", "weather", "track_condition", "track_condition_category",
        "venue_code",
    ]].copy()

    # --- Summary -------------------------------------------------------------
    print()
    print("Signal Summary")
    print("-" * 40)
    print(f"  Total races       : {len(out):,}")
    print(f"  Date range        : {out['meet_date'].min().date()} → {out['meet_date'].max().date()}")
    print(f"  Entropy bin counts:")
    for label, cnt in out["entropy_bin"].value_counts().sort_index().items():
        print(f"    {label:>8s}: {cnt:,}")
    print(f"  Venues: {out['venue'].nunique()}")

    # --- Write ---------------------------------------------------------------
    out_path = PROCESSED_DIR + "/entropy_signal.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWritten → {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
