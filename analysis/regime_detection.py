#!/usr/bin/env python3
"""Regime detection using CUSUM and wavelet-based change point analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt

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
# CUSUM change point detection
# ---------------------------------------------------------------------------

def cusum_changepoints(signal, threshold=None):
    """Detect change points using CUSUM on the signal."""
    x = np.array(signal, dtype=float)
    n = len(x)
    mu = x.mean()
    if threshold is None:
        threshold = 2.0 * x.std()

    s_pos = np.zeros(n)
    s_neg = np.zeros(n)
    changepoints = []

    for i in range(1, n):
        s_pos[i] = max(0, s_pos[i - 1] + (x[i] - mu))
        s_neg[i] = min(0, s_neg[i - 1] + (x[i] - mu))
        if s_pos[i] > threshold:
            changepoints.append((i, "positive"))
            s_pos[i] = 0
        if s_neg[i] < -threshold:
            changepoints.append((i, "negative"))
            s_neg[i] = 0

    return changepoints, s_pos, s_neg


# ---------------------------------------------------------------------------
# Wavelet-based change point detection
# ---------------------------------------------------------------------------

def wavelet_changepoints(signal, wavelet="mexh", n_top=10):
    """Detect change points using CWT with Mexican hat wavelet."""
    x = np.array(signal, dtype=float)
    scales = np.arange(1, 128)
    coeffs, freqs = pywt.cwt(x, scales, wavelet)
    # Sum absolute coefficients across scales for each time point
    power = np.sum(np.abs(coeffs), axis=0)
    # Find peaks in the aggregated power
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(power, distance=50, prominence=np.std(power))
    # Sort by prominence and take top n
    if len(peaks) > 0 and "prominences" in properties:
        top_idx = np.argsort(properties["prominences"])[::-1][:n_top]
        top_peaks = peaks[top_idx]
    else:
        top_peaks = peaks[:n_top]
    return sorted(top_peaks), coeffs, scales


def main():
    """Run regime detection analysis."""
    print("=" * 60)
    print("ANALYSIS: Regime Detection")
    print("=" * 60)

    df = pd.read_csv(PROCESSED_DIR + "/entropy_signal.csv", parse_dates=["meet_date"])
    print(f"  Loaded {len(df):,} races")

    signal = df["entropy_H"].values
    seq_ids = df["race_seq_id"].values
    dates = df["meet_date"].values

    # ---- CUSUM change points -------------------------------------------------
    print("  Running CUSUM change point detection...")
    cusum_cps, s_pos, s_neg = cusum_changepoints(signal)
    print(f"  CUSUM detected {len(cusum_cps)} change points")

    # ---- Wavelet change points -----------------------------------------------
    print("  Running wavelet change point detection...")
    wav_cps, coeffs, scales = wavelet_changepoints(signal, n_top=10)
    print(f"  Wavelet detected {len(wav_cps)} change points")

    # Combine unique change points
    all_cp_indices = sorted(set([cp[0] for cp in cusum_cps] + list(wav_cps)))

    # ---- Figure 1: Entropy signal with change points -------------------------
    fig1, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 1]})

    # Panel 1: Signal + change points
    ax = axes[0]
    ax.plot(seq_ids, signal, lw=0.3, alpha=0.5, color="steelblue")
    # Rolling mean overlay
    window = 100
    rolling_mean = pd.Series(signal).rolling(window, center=True).mean()
    ax.plot(seq_ids, rolling_mean, lw=1.5, color="navy", label=f"{window}-race rolling mean")

    # Mark change points
    for idx in wav_cps:
        if idx < len(seq_ids):
            ax.axvline(seq_ids[idx], color="red", alpha=0.5, lw=0.8)
    for cp_idx, cp_type in cusum_cps[:20]:
        if cp_idx < len(seq_ids):
            color = "green" if cp_type == "positive" else "orange"
            ax.axvline(seq_ids[cp_idx], color=color, alpha=0.3, lw=0.6, ls=":")

    ax.set_ylabel("Shannon Entropy H (bits)")
    ax.set_title("Entropy Signal with Detected Change Points")
    ax.legend(loc="upper right")

    # Panel 2: CUSUM positive
    axes[1].plot(seq_ids, s_pos, lw=0.5, color="green", label="CUSUM S+")
    axes[1].plot(seq_ids, s_neg, lw=0.5, color="orange", label="CUSUM S−")
    axes[1].set_ylabel("CUSUM statistic")
    axes[1].legend(loc="upper right", fontsize=8)

    # Panel 3: Wavelet power (summed across scales)
    power = np.sum(np.abs(coeffs), axis=0)
    axes[2].plot(seq_ids, power, lw=0.5, color="purple")
    for idx in wav_cps:
        if idx < len(seq_ids):
            axes[2].axvline(seq_ids[idx], color="red", alpha=0.5, lw=0.8)
    axes[2].set_ylabel("CWT Power")
    axes[2].set_xlabel("Race Sequence ID")

    fig1.tight_layout()
    save(fig1, "regime_changepoints")
    plt.close(fig1)

    # ---- Figure 2: Scalogram ------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(16, 6))
    # Limit to manageable display
    im = ax2.pcolormesh(seq_ids, scales, np.abs(coeffs),
                        cmap="magma", shading="auto")
    for idx in wav_cps:
        if idx < len(seq_ids):
            ax2.axvline(seq_ids[idx], color="cyan", alpha=0.6, lw=0.8, ls="--")
    ax2.set_xlabel("Race Sequence ID")
    ax2.set_ylabel("CWT Scale")
    ax2.set_title("Continuous Wavelet Transform Scalogram (Mexican Hat)")
    fig2.colorbar(im, ax=ax2, label="|CWT Coefficient|")
    save(fig2, "regime_scalogram")
    plt.close(fig2)

    # ---- Print change point list ---------------------------------------------
    print()
    print("Detected Change Points (Wavelet, top 10)")
    print("-" * 60)
    print(f"  {'Rank':>4s}  {'SeqID':>6s}  {'Date':>12s}  {'Entropy':>8s}")
    for rank, idx in enumerate(wav_cps, 1):
        if idx < len(df):
            row = df.iloc[idx]
            d = pd.Timestamp(row["meet_date"]).strftime("%Y-%m-%d")
            print(f"  {rank:>4d}  {int(row['race_seq_id']):>6d}  {d:>12s}  {row['entropy_H']:8.4f}")

    print()
    print(f"Total CUSUM change points: {len(cusum_cps)}")
    print(f"Total Wavelet change points: {len(wav_cps)}")


if __name__ == "__main__":
    main()
