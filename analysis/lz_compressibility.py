#!/usr/bin/env python3
"""Lempel-Ziv 76 compressibility analysis of entropy signal."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


# ---------------------------------------------------------------------------
# Lempel-Ziv 76 complexity
# ---------------------------------------------------------------------------

def lz76_complexity_raw(sequence):
    """Compute raw LZ76 word count c(s) using exhaustive history parsing."""
    s = list(sequence)
    n = len(s)
    if n == 0:
        return 0
    words = 0
    i = 0
    while i < n:
        # Find the shortest substring s[i:i+l+1] not seen in s[0:i+l]
        length = 1
        found = True
        while found and (i + length) <= n:
            substr = s[i: i + length]
            # Search in history s[0 : i + length - 1]
            history_end = i + length - 1
            found_in_history = False
            for j in range(history_end):
                if s[j: j + length] == substr and j + length <= history_end:
                    found_in_history = True
                    break
            if found_in_history:
                length += 1
                found = True
            else:
                found = False
        words += 1
        i += length
    return words


def lz76_complexity_normalised(sequence):
    """Compute normalised LZ76 complexity C(s) = c(s) * log2(n) / n."""
    n = len(sequence)
    if n == 0:
        return 0.0
    c = lz76_complexity_raw(sequence)
    return c * np.log2(n) / n


def main():
    """Run LZ76 compressibility analysis."""
    print("=" * 60)
    print("ANALYSIS: Lempel-Ziv Compressibility")
    print("=" * 60)

    df = pd.read_csv(PROCESSED_DIR + "/entropy_signal.csv", parse_dates=["meet_date"])
    print(f"  Loaded {len(df):,} races")

    # Convert entropy_bin to symbol sequence
    symbol_map = {"low": 0, "medium": 1, "high": 2}
    symbols = df["entropy_bin"].map(symbol_map).dropna().astype(int).tolist()
    n = len(symbols)
    print(f"  Symbol sequence length: {n}")

    # ---- Compute real LZ76 complexity ----------------------------------------
    print("  Computing LZ76 for real sequence...")
    real_c = lz76_complexity_raw(symbols)
    real_C = real_c * np.log2(n) / n
    print(f"  Real LZ76: c={real_c}, C={real_C:.6f}")

    # ---- Null distribution (1000 shuffles) -----------------------------------
    print("  Computing null distribution (1000 shuffles)...")
    rng = np.random.default_rng(42)
    null_C = []
    for i in range(1000):
        shuffled = list(symbols)
        rng.shuffle(shuffled)
        c_shuf = lz76_complexity_raw(shuffled)
        null_C.append(c_shuf * np.log2(n) / n)
        if (i + 1) % 200 == 0:
            print(f"    ... {i + 1}/1000 done")

    null_C = np.array(null_C)
    z_score = (real_C - null_C.mean()) / null_C.std()
    # One-sided p-value: proportion of null values <= real value
    p_value = np.mean(null_C <= real_C)

    print(f"  Null mean: {null_C.mean():.6f}, std: {null_C.std():.6f}")
    print(f"  z-score: {z_score:.3f}")
    print(f"  p-value (one-sided, lower tail): {p_value:.4f}")

    if z_score < -1.96:
        interpretation = "SIGNIFICANT: sequence is MORE compressible than random → detectable structure → possible market inefficiency"
    elif z_score > 1.96:
        interpretation = "SIGNIFICANT: sequence is LESS compressible than random → unexpected"
    else:
        interpretation = "NOT SIGNIFICANT: sequence compressibility is consistent with random"
    print(f"  Interpretation: {interpretation}")

    # ---- Figure 1: Null distribution + real value ----------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.hist(null_C, bins=40, color="steelblue", alpha=0.7, edgecolor="white",
             label="Null (shuffled)")
    ax1.axvline(real_C, color="red", lw=2, ls="--",
                label=f"Real C = {real_C:.4f}")
    ax1.axvline(null_C.mean(), color="gray", lw=1, ls=":",
                label=f"Null mean = {null_C.mean():.4f}")
    ax1.set_xlabel("Normalised LZ76 Complexity C(s)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"LZ76 Compressibility: z = {z_score:.2f}, p = {p_value:.4f}")
    ax1.legend()
    save(fig1, "lz76_null_distribution")
    plt.close(fig1)

    # ---- Figure 2: Rolling LZ76 over time -----------------------------------
    print("  Computing rolling LZ76 (window=200, step=20)...")
    window, step = 200, 20
    positions, rolling_c = [], []
    for start in range(0, n - window, step):
        chunk = symbols[start: start + window]
        c_val = lz76_complexity_raw(chunk)
        C_val = c_val * np.log2(window) / window
        positions.append(start + window // 2)
        rolling_c.append(C_val)

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.plot(positions, rolling_c, color="steelblue", lw=1)
    ax2.axhline(null_C.mean(), color="red", ls="--", lw=1,
                label=f"Random baseline = {null_C.mean():.4f}")
    ax2.set_xlabel("Race Sequence Position")
    ax2.set_ylabel("Normalised LZ76 Complexity C(s)")
    ax2.set_title("Rolling LZ76 Complexity Over Time (window=200)")
    ax2.legend()
    save(fig2, "lz76_rolling")
    plt.close(fig2)

    # ---- Figure 3: LZ76 by venue --------------------------------------------
    print("  Computing LZ76 by venue...")
    venue_results = []
    for v in sorted(df["venue"].dropna().unique()):
        vdf = df[df["venue"] == v]
        v_syms = vdf["entropy_bin"].map(symbol_map).dropna().astype(int).tolist()
        if len(v_syms) < 30:
            continue
        v_c = lz76_complexity_raw(v_syms)
        v_C = v_c * np.log2(len(v_syms)) / len(v_syms)
        venue_results.append({"venue": v, "n": len(v_syms), "C": v_C})

    vr = pd.DataFrame(venue_results).sort_values("C")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.barh(vr["venue"], vr["C"], color="teal", edgecolor="white")
    ax3.axvline(null_C.mean(), color="red", ls="--", lw=1,
                label=f"Random baseline = {null_C.mean():.4f}")
    ax3.set_xlabel("Normalised LZ76 Complexity")
    ax3.set_title("LZ76 Complexity by Venue")
    ax3.legend()
    save(fig3, "lz76_by_venue")
    plt.close(fig3)

    print()
    print("LZ76 by Venue")
    print("-" * 40)
    for _, row in vr.iterrows():
        print(f"  {row['venue']:>12s}: C={row['C']:.4f}  n={row['n']:,}")


if __name__ == "__main__":
    main()
