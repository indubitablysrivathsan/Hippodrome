#!/usr/bin/env python3
"""Build race-level features including Shannon entropy from entries_market.csv."""

import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def shannon_entropy(probs):
    """Compute Shannon entropy in bits (base 2) for a probability array."""
    probs = np.array(probs, dtype=float)
    probs = probs[probs > 0]  # Remove zeros to avoid log(0)
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))


def normalise_weather(w):
    """Normalise weather string to a standard category."""
    if pd.isna(w):
        return "Unknown"
    s = str(w).strip().upper()
    if s in ("CLOUDY",):
        return "Cloudy"
    if s in ("SUNNY", "FINE"):
        return "Fine"
    if s in ("OVERCAST",):
        return "Overcast"
    if s in ("RAIN", "RAINY", "RAINING"):
        return "Rainy"
    if s in ("HOT",):
        return "Hot"
    # Fallback: title case the original
    return str(w).strip().title()


def normalise_track_condition(tc):
    """Normalise track condition string to a standard text."""
    if pd.isna(tc):
        return "Unknown"
    return str(tc).strip().title()


def track_condition_category(tc):
    """Classify track condition into simplified categories."""
    if pd.isna(tc):
        return "Unknown"
    s = str(tc).upper()
    if "GOOD" in s or "FIRM" in s:
        return "Good"
    if "SOFT" in s or "YIELDING" in s:
        return "Soft"
    if "HEAVY" in s:
        return "Heavy"
    return "Unknown"


def extract_penetrometer(val):
    """Extract numeric penetrometer value from possibly messy string."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # Take the first numeric-looking portion
    parts = s.split()
    for p in parts:
        try:
            return float(p)
        except ValueError:
            continue
    return np.nan


def parse_first_margin(margins_str):
    """Extract the first margin from a comma-separated margin string."""
    if pd.isna(margins_str):
        return np.nan
    s = str(margins_str).strip()
    if not s:
        return np.nan
    first = s.split(",")[0].strip()
    return first


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Build race features."""
    print("=" * 60)
    print("PIPELINE STEP 2: Build Race Features")
    print("=" * 60)

    # --- Load data -----------------------------------------------------------
    entries_path = PROCESSED_DIR + "/entries_market.csv"
    races_path = RAW_DIR + "/races.csv"
    meetings_path = RAW_DIR + "/meetings.csv"

    entries = pd.read_csv(entries_path, parse_dates=["meet_date"])
    races = pd.read_csv(races_path)
    meetings = pd.read_csv(meetings_path)
    print(f"  Entries loaded : {len(entries):,} rows")
    print(f"  Races loaded   : {len(races):,} rows")
    print(f"  Meetings loaded: {len(meetings):,} rows")

    # --- Parse dates ---------------------------------------------------------
    races["meet_date"] = pd.to_datetime(races["meet_date"], errors="coerce")
    meetings["meet_date"] = pd.to_datetime(meetings["meet_date"], errors="coerce")

    # Normalise venues
    races["venue"] = races["venue"].apply(lambda v: str(v).strip().title() if pd.notna(v) else "Unknown")
    meetings["venue"] = meetings["venue"].apply(lambda v: str(v).strip().title() if pd.notna(v) else "Unknown")

    # --- Compute per-race aggregates from entries ----------------------------
    race_key = ["meet_date", "venue", "race_no"]

    agg = entries.groupby(race_key).agg(
        field_size=("implied_prob_norm", "size"),
        overround=("implied_prob_raw", "sum"),
        entropy_H=("implied_prob_norm", lambda x: shannon_entropy(x.values)),
        winner_prob=("implied_prob_norm", lambda x: x[entries.loc[x.index, "is_winner"]].values[0]
                     if entries.loc[x.index, "is_winner"].any() else np.nan),
        winner_was_favourite=(
            "prob_rank",
            lambda x: bool(
                (entries.loc[x.index].loc[entries.loc[x.index, "is_winner"], "prob_rank"] == 1).any()
            ),
        ),
        winner_prob_rank=(
            "prob_rank",
            lambda x: entries.loc[x.index].loc[entries.loc[x.index, "is_winner"], "prob_rank"].values[0]
            if entries.loc[x.index, "is_winner"].any() else np.nan,
        ),
        max_upset=("upset_magnitude", "max"),
    ).reset_index()

    # Overround is sum of raw implied probs minus 1
    agg["overround"] = agg["overround"] - 1.0

    # Normalised entropy
    agg["entropy_H_norm"] = agg.apply(
        lambda row: row["entropy_H"] / np.log2(row["field_size"])
        if row["field_size"] > 1 else 0.0,
        axis=1,
    )

    print(f"  Computed features for {len(agg):,} races")

    # --- Merge race metadata -------------------------------------------------
    races_meta = races[
        ["meet_date", "venue", "race_no", "race_name", "class_conditions",
         "distance_meters", "margins"]
    ].copy()
    agg = agg.merge(races_meta, on=race_key, how="left")

    # Winner margin raw (first margin value)
    agg["winner_margin_raw"] = agg["margins"].apply(parse_first_margin)
    agg.drop(columns=["margins"], inplace=True)

    # --- Merge meeting metadata ----------------------------------------------
    meetings_meta = meetings[
        ["meet_date", "venue", "season", "weather", "track_condition", "penetrometer"]
    ].copy()
    meetings_meta["weather"] = meetings_meta["weather"].apply(normalise_weather)
    meetings_meta["track_condition_norm"] = meetings_meta["track_condition"].apply(normalise_track_condition)
    meetings_meta["track_condition_category"] = meetings_meta["track_condition"].apply(track_condition_category)
    meetings_meta["penetrometer_num"] = meetings_meta["penetrometer"].apply(extract_penetrometer)

    agg = agg.merge(meetings_meta, on=["meet_date", "venue"], how="left")

    # Rename / clean columns
    agg.rename(columns={
        "track_condition_norm": "track_condition_clean",
        "penetrometer_num": "penetrometer_val",
    }, inplace=True)

    # --- Assemble output -----------------------------------------------------
    out = agg[[
        "meet_date", "venue", "race_no", "race_name", "class_conditions",
        "distance_meters", "field_size", "overround",
        "entropy_H", "entropy_H_norm",
        "winner_prob", "winner_was_favourite", "winner_prob_rank", "max_upset",
        "winner_margin_raw",
        "season", "weather", "track_condition_clean", "track_condition_category",
        "penetrometer_val",
    ]].copy()
    out.rename(columns={
        "track_condition_clean": "track_condition",
        "penetrometer_val": "penetrometer",
    }, inplace=True)

    # --- Summary stats -------------------------------------------------------
    print()
    print("Entropy Statistics")
    print("-" * 40)
    print(f"  Mean H     : {out['entropy_H'].mean():.4f} bits")
    print(f"  Std H      : {out['entropy_H'].std():.4f}")
    print(f"  Min H      : {out['entropy_H'].min():.4f}")
    print(f"  Max H      : {out['entropy_H'].max():.4f}")
    print(f"  Mean H_norm: {out['entropy_H_norm'].mean():.4f}")
    print()
    print("Overround Statistics")
    print("-" * 40)
    print(f"  Mean       : {out['overround'].mean():.4f}")
    print(f"  Std        : {out['overround'].std():.4f}")
    print(f"  Min        : {out['overround'].min():.4f}")
    print(f"  Max        : {out['overround'].max():.4f}")
    print()
    fav_wins = out["winner_was_favourite"].sum()
    total = len(out)
    print(f"Favourite wins: {fav_wins:,} / {total:,} = {fav_wins / total:.2%}")

    # --- Write ---------------------------------------------------------------
    out_path = PROCESSED_DIR + "/race_features.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWritten → {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
