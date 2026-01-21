"""
This script:
1) Detects thumb–index grasp events using the Manus pinch signal (Pinch_ThumbToIndex)
   via a double-threshold method.
2) Computes percentile-based ROM metrics (P95–P5) for:
   - Thumb_CMC_Flex
   - Index_MCP_Flex
3) Exports:
   - a PNG figure (pinch signal + events + thresholds)
   - a CSV summary (counts, durations, ROM, thresholds)

"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Expected CSV columns
# -------------------------

TIME_CANDIDATES = ["Elapsed_Time_In_Milliseconds", "Time", "Frame"]
PINCH_COL = "Pinch_ThumbToIndex"

# Joint angle channels (degrees) from Manus Core export
THUMB_COL = "Thumb_CMC_Flex"
INDEX_COL = "Index_MCP_Flex"


def build_time_seconds(df: pd.DataFrame) -> np.ndarray:
    # Build a time vector in seconds from available columns
    for c in TIME_CANDIDATES:
        if c in df.columns:
            t = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            if np.all(~np.isfinite(t)):
                continue
            if c == "Elapsed_Time_In_Milliseconds":
                return t / 1000.0
            return t

    # Fallback: sample index as "time"
    return np.arange(len(df), dtype=float)


def rolling_median(x: np.ndarray, win: int = 9) -> np.ndarray:
    # Smoothing to reduce noise in pinch signal
    if win <= 1:
        return x
    return pd.Series(x).rolling(win, center=True, min_periods=1).median().to_numpy()


def detect_pinch_events(t_s, pinch, close_q=0.15, open_q=0.35,
                        smooth_win=9, min_gap_s=0.20):
    """
    Detect grasp events from pinch signal using a double-threshold approach.

    Adaptive thresholds:
    - Compute q10 and q90 of smoothed pinch signal
    - Define amplitude span = q90 - q10
    - close_thr = q10 + close_q * span
    - open_thr  = q10 + open_q  * span  (must be > close_thr)

    A grasp is defined as a close event followed by the next open event.
    min_gap_s prevents multiple detections within a very short time.

    Returns:
        pinch_s: smoothed pinch
        close_thr, open_thr: thresholds
        paired_close_idx, paired_open_idx: matched close/open indices
    """
    pinch_s = rolling_median(pinch, win=smooth_win)

    q10 = np.nanquantile(pinch_s, 0.10)
    q90 = np.nanquantile(pinch_s, 0.90)
    span = q90 - q10 if np.isfinite(q90 - q10) and (q90 - q10) > 1e-9 else np.nanstd(pinch_s)

    close_thr = q10 + close_q * span
    open_thr = q10 + open_q * span

    # Fallback if thresholds are invalid
    if not np.isfinite(close_thr) or not np.isfinite(open_thr) or open_thr <= close_thr:
        med = np.nanmedian(pinch_s)
        sd = np.nanstd(pinch_s)
        close_thr = med - 0.8 * sd
        open_thr = med - 0.2 * sd

    close_idx, open_idx = [], []
    closed = False
    last_event_t = -np.inf

    for i in range(len(pinch_s)):
        if not np.isfinite(pinch_s[i]) or not np.isfinite(t_s[i]):
            continue

        if (not closed) and pinch_s[i] < close_thr and (t_s[i] - last_event_t >= min_gap_s):
            closed = True
            close_idx.append(i)
            last_event_t = t_s[i]

        elif closed and pinch_s[i] > open_thr and (t_s[i] - last_event_t >= min_gap_s):
            closed = False
            open_idx.append(i)
            last_event_t = t_s[i]

    close_idx = np.array(close_idx, dtype=int)
    open_idx = np.array(open_idx, dtype=int)

    # Pair each close with the first open that occurs after it
    paired_close, paired_open = [], []
    j = 0
    for ci in close_idx:
        while j < len(open_idx) and open_idx[j] <= ci:
            j += 1
        if j < len(open_idx):
            paired_close.append(ci)
            paired_open.append(open_idx[j])
            j += 1

    return pinch_s, close_thr, open_thr, np.array(paired_close), np.array(paired_open)


def rom_p95_p5(x: np.ndarray) -> float:
    #Percentile-based ROM (P95 - P5) to reduce outlier sensitivity
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return float("nan")
    return float(np.percentile(x, 95) - np.percentile(x, 5))


def main(csv_path, out_dir, show_plots=True):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    t = build_time_seconds(df)

    pinch = pd.to_numeric(df[PINCH_COL], errors="coerce").to_numpy(dtype=float)
    thumb = pd.to_numeric(df[THUMB_COL], errors="coerce").to_numpy(dtype=float)
    index = pd.to_numeric(df[INDEX_COL], errors="coerce").to_numpy(dtype=float)

    thumb_rom = rom_p95_p5(thumb)
    index_rom = rom_p95_p5(index)

    pinch_s, close_thr, open_thr, close_idx, open_idx = detect_pinch_events(t, pinch)

    grasp_durs = (t[open_idx] - t[close_idx]) if len(close_idx) else np.array([])
    grasp_durs = grasp_durs[np.isfinite(grasp_durs) & (grasp_durs > 0)]

    # =========================
    # FIGURE: Pinch + events
    # =========================
    fig = plt.figure(figsize=(10, 4))
    plt.plot(t, pinch_s, label="Pinch (smoothed)")
    plt.axhline(close_thr, linestyle="--", label="close_thr")
    plt.axhline(open_thr, linestyle="--", label="open_thr")

    if len(close_idx):
        plt.scatter(t[close_idx], pinch_s[close_idx], marker="o", label="close")
    if len(open_idx):
        plt.scatter(t[open_idx], pinch_s[open_idx], marker="x", label="open")

    plt.xlabel("Time (s)")
    plt.ylabel(PINCH_COL)
    plt.title(f"{csv_path.stem} | Pinch events | ROM thumb={thumb_rom:.2f}°, ROM index={index_rom:.2f}°")
    plt.legend()
    plt.tight_layout()

    out_png = out_dir / f"{csv_path.stem}__pinch_events.png"
    fig.savefig(out_png, dpi=200)

    # ---CSV Summary
    summary = pd.DataFrame([{
        "file": str(csv_path),
        "duration_s": float(np.nanmax(t) - np.nanmin(t)),
        "n_grasps": int(len(grasp_durs)),
        "grasp_duration_mean_s": float(np.nanmean(grasp_durs)) if len(grasp_durs) else np.nan,
        "grasp_duration_median_s": float(np.nanmedian(grasp_durs)) if len(grasp_durs) else np.nan,
        "thumb_rom_p95_p5_deg": thumb_rom,
        "index_rom_p95_p5_deg": index_rom,
        "close_thr": float(close_thr),
        "open_thr": float(open_thr),
    }])

    out_csv = out_dir / f"{csv_path.stem}__summary.csv"
    summary.to_csv(out_csv, index=False)

    print("Saved:", out_png)
    print("Saved:", out_csv)
    print(summary.to_string(index=False))

    if show_plots:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main(
        csv_path=r"C:\Users\jeann\Documents\MetaGlove_Project\data\BBTRH60-002_CIIRC_R.csv",
        out_dir=r"C:\Users\jeann\Documents\MetaGlove_Project\output",
        show_plots=True
    )
