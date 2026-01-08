"""
Metaglove CSV analysis for NHPT / BBT (Manus / Xsens)

This script:
- Loads a Metaglove CSV file (NHPT or BBT)
- Builds a time vector in seconds (t_s) + estimates sampling rate (fs_hz)
- Infers test type from filename: "NHPT" / "BBT"
- Detects pinch close/open events using hysteresis thresholds
- Computes metrics:
   * number of grasps, grasp durations
   * cycle durations (close->close) + variability (CV)
   * joint angles ROM (P95 - P5)
- Saves plots

"""


from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# CONFIG
# -----------------------------


DEFAULT_PINCH_COL = "Pinch_ThumbToIndex"  # usually the most useful for NHPT/BBT


ANGLE_COLS_CANDIDATES = [
   # Thumb
   "Thumb_CMC_Flex", "Thumb_CMC_Spread", "Thumb_PIP_Flex", "Thumb_DIP_Flex",
   # Index
   "Index_MCP_Flex", "Index_MCP_Spread", "Index_PIP_Flex", "Index_DIP_Flex",
   # Middle
   "Middle_MCP_Flex", "Middle_MCP_Spread", "Middle_PIP_Flex", "Middle_DIP_Flex",
   # Ring
   "Ring_MCP_Flex", "Ring_MCP_Spread", "Ring_PIP_Flex", "Ring_DIP_Flex",
   # Pinky
   "Pinky_MCP_Flex", "Pinky_MCP_Spread", "Pinky_PIP_Flex", "Pinky_DIP_Flex",
]


# -----------------------------
# TEST TYPE INFERENCE
# -----------------------------


def infer_test_type_from_filename(csv_file: Path) -> str:
   """
   Infer test type from filename (stem).
   Expected patterns contain 'NHPT' or 'BBT' in the name.


   Returns:
       "NHPT", "BBT", or "UNKNOWN"
   """
   name = csv_file.stem.upper()
   if "NHPT" in name:
       return "NHPT"
   if "BBT" in name:
       return "BBT"
   return "UNKNOWN"


# -----------------------------
# DATA STRUCTURES
# -----------------------------


@dataclass
class PinchEvents:
   """Detected pinch close/open events."""
   close_thr: float
   open_thr: float
   close_times_s: np.ndarray
   open_times_s: np.ndarray
   close_idx: np.ndarray
   open_idx: np.ndarray




@dataclass
class Summary:
   """Per-file summary metrics."""
   file: str
   file_stem: str
   test_type: str


   n_samples: int
   duration_s: float
   fs_hz: float


   pinch_col: str
   pinch_min: float
   pinch_mean: float
   pinch_p10: float
   pinch_p90: float
   pinch_close_thr: float
   pinch_open_thr: float


   n_grasps: int
   grasp_duration_mean_s: float
   grasp_duration_median_s: float


   cycle_mean_s: float
   cycle_cv: float  # coefficient of variation


   inter_grasp_interval_mean_s: float
   inter_grasp_interval_median_s: float


   rom_mean: float
   rom_details: str  # compact text for traceability




# -----------------------------
# HELPERS
# -----------------------------


def to_numeric_df(df: pd.DataFrame, min_non_nan_ratio: float = 0.2) -> pd.DataFrame:
   """
   Convert object columns to numeric when it makes sense (no errors='ignore').


   We only replace a column if conversion yields enough numeric values.


   Args:
       df: input dataframe
       min_non_nan_ratio: minimal fraction of numeric (non-NaN) values after conversion
                          to accept replacement.


   Returns:
       dataframe with selected columns converted to numeric
   """
   out = df.copy()
   for c in out.columns:
       if out[c].dtype == object:
           conv = pd.to_numeric(out[c], errors="coerce")
           ratio = float(np.isfinite(conv.to_numpy(dtype=float)).mean())
           if ratio >= min_non_nan_ratio:
               out[c] = conv
   return out




def build_time_seconds(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
   """
   Build a time column in seconds (t_s) and estimate sampling frequency (fs_hz).


   Priority for time:
   1) Elapsed_Time_In_Milliseconds (ms -> s)
   2) Time (numeric)
   3) Frame (numeric)
   4) index


   Returns:
       df (with t_s), fs_hz
   """
   df = df.copy()


   if "Elapsed_Time_In_Milliseconds" in df.columns:
       t_s = pd.to_numeric(df["Elapsed_Time_In_Milliseconds"], errors="coerce") / 1000.0
       if t_s.isna().all():
           raise ValueError("Elapsed_Time_In_Milliseconds exists but is not usable (all NaN).")
       df["t_s"] = t_s


   elif "Time" in df.columns:
       t = pd.to_numeric(df["Time"], errors="coerce")
       if t.isna().all():
           raise ValueError("No usable time column (Time is not numeric).")
       df["t_s"] = t


   elif "Frame" in df.columns:
       df["t_s"] = pd.to_numeric(df["Frame"], errors="coerce")


   else:
       df["t_s"] = np.arange(len(df), dtype=float)


   t = df["t_s"].to_numpy(dtype=float)


   dt = np.diff(t)
   dt = dt[np.isfinite(dt) & (dt > 0)]
   fs_hz = float(1.0 / np.median(dt)) if len(dt) else 0.0


   # If time was actually in milliseconds, fix it
   if fs_hz > 500:
       df["t_s"] = df["t_s"] / 1000.0
       t = df["t_s"].to_numpy(dtype=float)
       dt = np.diff(t)
       dt = dt[np.isfinite(dt) & (dt > 0)]
       fs_hz = float(1.0 / np.median(dt)) if len(dt) else 0.0


   return df, fs_hz




def smooth_series(x: np.ndarray, win: int = 9) -> np.ndarray:
   """Rolling median smoothing (helps stabilize event detection)."""
   if win <= 1:
       return x
   s = pd.Series(x)
   return s.rolling(window=win, center=True, min_periods=1).median().to_numpy()




def detect_pinch_events(
   t_s: np.ndarray,
   pinch: np.ndarray,
   close_q: float = 0.15,
   open_q: float = 0.35,
   smooth_win: int = 9,
   min_gap_s: float = 0.20,
) -> PinchEvents:
   """
   Detect pinch close/open events with hysteresis:
   - closed state when pinch < close_thr
   - open state when pinch > open_thr


   Thresholds are computed from quantiles for robustness across participants/files.
   """
   pinch = pinch.astype(float)
   pinch_s = smooth_series(pinch, win=smooth_win)


   q10 = np.nanquantile(pinch_s, 0.10)
   q90 = np.nanquantile(pinch_s, 0.90)
   span = (q90 - q10) if np.isfinite(q90 - q10) and (q90 - q10) > 1e-9 else np.nanstd(pinch_s)


   close_thr = q10 + close_q * span
   open_thr = q10 + open_q * span


   # Fallback if thresholds are not usable
   if (not np.isfinite(close_thr)) or (not np.isfinite(open_thr)) or (open_thr <= close_thr):
       med = np.nanmedian(pinch_s)
       sd = np.nanstd(pinch_s)
       close_thr = med - 0.8 * sd
       open_thr = med - 0.2 * sd


   closed = False
   close_idx: List[int] = []
   open_idx: List[int] = []
   last_event_t = -np.inf


   for i in range(len(pinch_s)):
       if not np.isfinite(pinch_s[i]) or not np.isfinite(t_s[i]):
           continue


       if (not closed) and (pinch_s[i] < close_thr):
           if (t_s[i] - last_event_t) >= min_gap_s:
               closed = True
               close_idx.append(i)
               last_event_t = t_s[i]


       elif closed and (pinch_s[i] > open_thr):
           if (t_s[i] - last_event_t) >= min_gap_s:
               closed = False
               open_idx.append(i)
               last_event_t = t_s[i]


   close_idx = np.array(close_idx, dtype=int)
   open_idx = np.array(open_idx, dtype=int)


   # Pair close -> next open
   paired_close: List[int] = []
   paired_open: List[int] = []
   j = 0
   for ci in close_idx:
       while j < len(open_idx) and open_idx[j] <= ci:
           j += 1
       if j < len(open_idx):
           paired_close.append(ci)
           paired_open.append(open_idx[j])
           j += 1


   close_idx = np.array(paired_close, dtype=int)
   open_idx = np.array(paired_open, dtype=int)


   return PinchEvents(
       close_thr=float(close_thr),
       open_thr=float(open_thr),
       close_times_s=t_s[close_idx] if len(close_idx) else np.array([], dtype=float),
       open_times_s=t_s[open_idx] if len(open_idx) else np.array([], dtype=float),
       close_idx=close_idx,
       open_idx=open_idx,
   )




def compute_rom(df: pd.DataFrame, angle_cols: List[str]) -> Tuple[float, Dict[str, float]]:
   """
   joint range-of-motion (ROM) using percentiles:
       ROM = P95 - P5


   Returns:
       rom_mean, rom_by_col
   """
   roms: Dict[str, float] = {}
   for c in angle_cols:
       if c not in df.columns:
           continue


       v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
       v = v[np.isfinite(v)]
       if len(v) < 50:
           continue


       p5 = float(np.nanpercentile(v, 5))
       p95 = float(np.nanpercentile(v, 95))
       roms[c] = p95 - p5


   if not roms:
       return float("nan"), {}


   rom_mean = float(np.nanmean(list(roms.values())))
   return rom_mean, roms




def safe_percentile(x: np.ndarray, q: float) -> float:
   """Percentile with NaN-safety."""
   x = x[np.isfinite(x)]
   if len(x) == 0:
       return float("nan")
   return float(np.percentile(x, q))




def compute_inter_grasp_intervals(events: PinchEvents) -> np.ndarray:
   """
   Inter-grasp interval = time between an 'open' event and the next 'close' event.
   This often captures pauses / transitions.
   """
   if len(events.open_times_s) == 0 or len(events.close_times_s) == 0:
       return np.array([], dtype=float)


   intervals: List[float] = []
   j = 0
   for ot in events.open_times_s:
       while j < len(events.close_times_s) and events.close_times_s[j] <= ot:
           j += 1
       if j < len(events.close_times_s):
           intervals.append(float(events.close_times_s[j] - ot))


   out = np.array(intervals, dtype=float)
   out = out[np.isfinite(out) & (out >= 0)]
   return out




# -----------------------------
# MAIN ANALYSIS
# -----------------------------


def analyze_file(
   csv_file: Path,
   out_dir: Path,
   pinch_col: str = DEFAULT_PINCH_COL,
   label: Optional[str] = None,
   show_plots: bool = False
) -> Summary:
   """Analyze a single Metaglove CSV file and export plots + summary CSV."""
   out_dir.mkdir(parents=True, exist_ok=True)
   test_type = infer_test_type_from_filename(csv_file)


   df = pd.read_csv(csv_file)
   df = to_numeric_df(df)
   df, fs_hz = build_time_seconds(df)


   t = df["t_s"].to_numpy(dtype=float)
   duration_s = float(np.nanmax(t) - np.nanmin(t)) if len(t) else 0.0


   if pinch_col not in df.columns:
       raise ValueError(f"Pinch column not found: {pinch_col}")


   pinch = pd.to_numeric(df[pinch_col], errors="coerce").to_numpy(dtype=float)
   pinch_s = smooth_series(pinch, win=9)


   events = detect_pinch_events(
       t_s=t,
       pinch=pinch,
       close_q=0.15,
       open_q=0.35,
       smooth_win=9,
       min_gap_s=0.20,
   )


   # Grasp durations (close -> open)
   grasp_durations = (events.open_times_s - events.close_times_s) if len(events.close_times_s) else np.array([])
   grasp_durations = grasp_durations[np.isfinite(grasp_durations) & (grasp_durations > 0)]


   # Cycle durations (close(i) -> close(i+1))
   if len(events.close_times_s) >= 2:
       cycles = np.diff(events.close_times_s)
       cycles = cycles[np.isfinite(cycles) & (cycles > 0)]
   else:
       cycles = np.array([])


   cycle_mean_s = float(np.nanmean(cycles)) if len(cycles) else float("nan")
   cycle_cv = (
       float(np.nanstd(cycles) / np.nanmean(cycles))
       if len(cycles) and np.nanmean(cycles) > 1e-9
       else float("nan")
   )


   # Inter-grasp intervals (open -> next close)
   inter_grasp = compute_inter_grasp_intervals(events)
   inter_grasp_mean_s = float(np.nanmean(inter_grasp)) if len(inter_grasp) else float("nan")
   inter_grasp_median_s = float(np.nanmedian(inter_grasp)) if len(inter_grasp) else float("nan")


   # Joint angles ROM
   present_angle_cols = [c for c in ANGLE_COLS_CANDIDATES if c in df.columns]
   rom_mean, roms = compute_rom(df, present_angle_cols)


   # -----------------------------
   # PLOTS
   # -----------------------------
   stem = csv_file.stem
   title = label if label else stem


   figs: List[plt.Figure] = []


   # Plot 1: Pinch signal + thresholds + events
   fig = plt.figure()
   plt.plot(t, pinch_s)
   plt.axhline(events.close_thr, linestyle="--")
   plt.axhline(events.open_thr, linestyle="--")
   if len(events.close_idx):
       plt.scatter(t[events.close_idx], pinch_s[events.close_idx], marker="o")
   if len(events.open_idx):
       plt.scatter(t[events.open_idx], pinch_s[events.open_idx], marker="x")
   plt.xlabel("Time (s)")
   plt.ylabel(pinch_col)
   plt.title(f"Pinch signal + events — {title} ({test_type})")
   plt.tight_layout()
   fig.savefig(out_dir / f"{stem}__pinch_events.png", dpi=200)
   figs.append(fig)


   # Plot 2: Cycle durations histogram (close -> close)
   fig = plt.figure()
   if len(cycles):
       plt.hist(cycles, bins=25)
   plt.xlabel("Cycle duration (s)  [close→close]")
   plt.ylabel("Count")
   plt.title(f"Cycle durations — {title} ({test_type})")
   plt.tight_layout()
   fig.savefig(out_dir / f"{stem}__cycle_hist.png", dpi=200)
   figs.append(fig)


   # Plot 3: Grasp duration histogram (close -> open)
   fig = plt.figure()
   if len(grasp_durations):
       plt.hist(grasp_durations, bins=25)
   plt.xlabel("Grasp duration (s)  [close→open]")
   plt.ylabel("Count")
   plt.title(f"Grasp durations — {title} ({test_type})")
   plt.tight_layout()
   fig.savefig(out_dir / f"{stem}__grasp_duration_hist.png", dpi=200)
   figs.append(fig)


   # Plot 4: Inter-grasp interval histogram (open -> next close)
   fig = plt.figure()
   if len(inter_grasp):
       plt.hist(inter_grasp, bins=25)
   plt.xlabel("Inter-grasp interval (s)  [open→next close]")
   plt.ylabel("Count")
   plt.title(f"Inter-grasp intervals — {title} ({test_type})")
   plt.tight_layout()
   fig.savefig(out_dir / f"{stem}__inter_grasp_interval_hist.png", dpi=200)
   figs.append(fig)


   # Plot 5: ROM bar plot (top joints)
   fig = plt.figure()
   if roms:
       rom_items = sorted(roms.items(), key=lambda kv: -kv[1])[:10]
       labels = [k for k, _ in rom_items]
       values = [v for _, v in rom_items]
       plt.bar(range(len(values)), values)
       plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
       plt.ylabel("ROM (P95 - P5)")
       plt.title(f"Joint ROM (top 10) — {title} ({test_type})")
       plt.tight_layout()
   else:
       plt.text(0.5, 0.5, "No angle columns found", ha="center", va="center")
       plt.axis("off")
       plt.title(f"Joint ROM — {title} ({test_type})")
       plt.tight_layout()
   fig.savefig(out_dir / f"{stem}__rom_top10.png", dpi=200)
   figs.append(fig)


   # Show all figures at once
   if show_plots:
       plt.show()
   else:
       for f in figs:
           plt.close(f)


   # -----------------------------
   # SUMMARY
   # -----------------------------
   pinch_min = float(np.nanmin(pinch_s)) if np.any(np.isfinite(pinch_s)) else float("nan")
   pinch_mean = float(np.nanmean(pinch_s)) if np.any(np.isfinite(pinch_s)) else float("nan")
   pinch_p10 = safe_percentile(pinch_s, 10)
   pinch_p90 = safe_percentile(pinch_s, 90)


   grasp_duration_mean_s = float(np.nanmean(grasp_durations)) if len(grasp_durations) else float("nan")
   grasp_duration_median_s = float(np.nanmedian(grasp_durations)) if len(grasp_durations) else float("nan")


   rom_items = sorted(roms.items(), key=lambda kv: -kv[1])[:10]
   rom_details = "; ".join([f"{k}:{v:.2f}" for k, v in rom_items])


   summary = Summary(
       file=str(csv_file),
       file_stem=stem,
       test_type=test_type,


       n_samples=int(len(df)),
       duration_s=duration_s,
       fs_hz=float(fs_hz),


       pinch_col=pinch_col,
       pinch_min=pinch_min,
       pinch_mean=pinch_mean,
       pinch_p10=pinch_p10,
       pinch_p90=pinch_p90,
       pinch_close_thr=float(events.close_thr),
       pinch_open_thr=float(events.open_thr),


       n_grasps=int(len(grasp_durations)),
       grasp_duration_mean_s=grasp_duration_mean_s,
       grasp_duration_median_s=grasp_duration_median_s,


       cycle_mean_s=cycle_mean_s,
       cycle_cv=cycle_cv,


       inter_grasp_interval_mean_s=inter_grasp_mean_s,
       inter_grasp_interval_median_s=inter_grasp_median_s,


       rom_mean=float(rom_mean),
       rom_details=rom_details,
   )


   pd.DataFrame([summary.__dict__]).to_csv(out_dir / f"{stem}__summary.csv", index=False)
   return summary




def analyze_many(
   files: List[Path],
   out_dir: Path,
   pinch_col: str = DEFAULT_PINCH_COL,
) -> pd.DataFrame:
   """Analyze multiple CSV files and export ALL_SUMMARIES.csv."""
   summaries: List[Dict] = []
   for f in files:
       try:
           s = analyze_file(f, out_dir=out_dir, pinch_col=pinch_col)
           summaries.append(s.__dict__)
       except Exception as e:
           summaries.append({
               "file": str(f),
               "file_stem": f.stem,
               "test_type": infer_test_type_from_filename(f),
               "error": str(e),
           })


   df_sum = pd.DataFrame(summaries)
   df_sum.to_csv(out_dir / "ALL_SUMMARIES.csv", index=False)
   return df_sum




# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
   csv_file = Path(r"C:\Users\jeann\Documents\MetaGlove_Project\data\BBTRH60-002 3_CIIRC_R.csv")
   out_dir = Path(r"C:\Users\jeann\Documents\MetaGlove_Project\output")


   summary = analyze_file(
       csv_file,
       out_dir=out_dir,
       pinch_col=DEFAULT_PINCH_COL,
       show_plots=True
   )


   print("==== SUMMARY ====")
   for k, v in summary.__dict__.items():
       print(f"{k}: {v}")