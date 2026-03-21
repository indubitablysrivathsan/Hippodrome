#!/usr/bin/env python3
"""Parse raw odds from runners.csv into decimal odds and implied probabilities."""

import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — works whether run from repo root or from inside pipeline/
# ---------------------------------------------------------------------------
RAW_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"

# ---------------------------------------------------------------------------
# Odds parsing
# ---------------------------------------------------------------------------

def parse_odds_to_decimal(odds_str):
    """Convert fractional odds string to decimal odds float."""
    if pd.isna(odds_str):
        return np.nan
    s = str(odds_str).strip().upper()
    if s in ("", "-"):
        return np.nan
    if s in ("EVS", "EVENS", "1/1"):
        return 2.0
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                num, den = float(parts[0]), float(parts[1])
                if den == 0:
                    return np.nan
                return 1 + num / den
            except ValueError:
                return np.nan
    try:
        # Could be a whole number like "6" meaning 6/1
        return 1 + float(s)
    except ValueError:
        return np.nan


def normalise_venue(v):
    """Normalise venue string to title case."""
    if pd.isna(v):
        return "Unknown"
    return str(v).strip().title()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    """Run odds parsing pipeline."""
    print("=" * 60)
    print("PIPELINE STEP 1: Parse Odds")
    print("=" * 60)

    # --- Load raw data -------------------------------------------------------
    runners_path = RAW_DIR + "/runners.csv"

    print(f"Reading {runners_path} ...")
    df = pd.read_csv(runners_path)
    print(f"  Raw runners rows: {len(df):,}")

    # --- Parse dates and normalise -------------------------------------------
    df["meet_date"] = pd.to_datetime(df["meet_date"], errors="coerce")
    df["venue"] = df["venue"].apply(normalise_venue)

    # --- Parse odds ----------------------------------------------------------
    df["odds_raw"] = df["odds"].astype(str)
    df["odds_decimal"] = df["odds"].apply(parse_odds_to_decimal)

    n_valid = df["odds_decimal"].notna().sum()
    n_invalid = df["odds_decimal"].isna().sum()
    print(f"  Valid odds parsed : {n_valid:,}")
    print(f"  Invalid / NaN odds: {n_invalid:,}")

    # --- Implied probability -------------------------------------------------
    df["implied_prob_raw"] = 1.0 / df["odds_decimal"]

    # --- Per-race valid runner count -----------------------------------------
    race_key = ["meet_date", "venue", "race_no"]
    valid_mask = df["odds_decimal"].notna()
    valid_counts = (
        df.loc[valid_mask]
        .groupby(race_key)
        .size()
        .reset_index(name="n_valid_runners")
    )
    df = df.merge(valid_counts, on=race_key, how="left")
    df["n_valid_runners"] = df["n_valid_runners"].fillna(0).astype(int)

    # Flag races with fewer than 2 valid runners
    bad_races = df["n_valid_runners"] < 2
    n_bad_race_rows = bad_races.sum()
    print(f"  Rows in races with <2 valid odds (excluded): {n_bad_race_rows:,}")
    df = df[~bad_races].copy()

    # Drop runners with NaN odds within good races
    df = df[df["odds_decimal"].notna()].copy()
    print(f"  Rows after cleanup: {len(df):,}")

    # --- Normalised probability per race -------------------------------------
    race_prob_sum = (
        df.groupby(race_key)["implied_prob_raw"]
        .transform("sum")
    )
    df["implied_prob_norm"] = df["implied_prob_raw"] / race_prob_sum

    # --- Probability rank (1 = favourite) ------------------------------------
    df["prob_rank"] = (
        df.groupby(race_key)["implied_prob_norm"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    # --- Placing cleanup -----------------------------------------------------
    df["placing"] = pd.to_numeric(df["placing"], errors="coerce")
    df["is_winner"] = df["placing"] == 1

    # --- Upset magnitude -----------------------------------------------------
    df["upset_magnitude"] = df["prob_rank"] - df["placing"]
    # positive = outperformed expectation (finished better than odds suggested)

    # --- Assemble output columns ---------------------------------------------
    out_cols = [
        "meet_date", "venue", "race_no", "horse_name", "horse_seq",
        "placing", "jockey", "trainer", "weight",
        "odds_raw", "odds_decimal", "implied_prob_raw", "implied_prob_norm",
        "prob_rank", "upset_magnitude", "is_winner",
    ]
    out = df[out_cols].copy()

    # --- Validation summary --------------------------------------------------
    n_races = out.groupby(race_key).ngroups
    overrounds = out.groupby(race_key)["implied_prob_raw"].sum() - 1
    print()
    print("Validation Summary")
    print("-" * 40)
    print(f"  Total runners in output : {len(out):,}")
    print(f"  Total races with data   : {n_races:,}")
    print(f"  Overround range         : {overrounds.min():.4f} – {overrounds.max():.4f}")
    print(f"  Mean overround          : {overrounds.mean():.4f}")
    print(f"  Favourite win rate      : {out.loc[out['is_winner'] & (out['prob_rank'] == 1)].shape[0] / out.loc[out['is_winner']].shape[0]:.3%}")

    # --- Write output --------------------------------------------------------
    out_path = PROCESSED_DIR + "/entries_market.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWritten → {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
