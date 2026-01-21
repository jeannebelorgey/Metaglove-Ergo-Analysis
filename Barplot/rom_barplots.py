"""
This script reads an Excel file containing ROM values across trials for each subject and hand:
- Thumb ROM trials: ROM thumb 1..3
- Index ROM trials: ROM index 1..3

It generates barplots (mean ± SD) and overlays individual trial points.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(r"C:\Users\jeann\Documents\MetaGlove_Project\Barplot")
XLSX_FILE = BASE_DIR / "ROM_data.xlsx"


# =========================
# Excel columns
# =========================
GROUP_COL = "Unnamed: 0"          #e.g., "subject 1 DH", "subject 1 NDH"
THUMB_COLS = ["ROM thumb 1", "ROM thumb 2", "ROM thumb 3"]
INDEX_COLS = ["ROM index 1", "ROM index 2", "ROM index 3"]


def parse_subject_and_hand(label: str):
    """Extract subject_id and hand (DH / NDH) from a row label."""
    if not isinstance(label, str):
        return None, None

    s = label.lower()

    #subject
    m = re.search(r"(subject)\s*(\d+)", s)
    subject_id = m.group(2) if m else "unknown"

    if "ndh" in s:
        hand = "NDH"
    elif "dh" in s:
        hand = "DH"
    else:
        hand = "unknown"

    return subject_id, hand


def mean_sd(x):
    """Return mean and sample SD"""
    if len(x) == 0:
        return np.nan, np.nan
    mean = np.mean(x)
    sd = np.std(x, ddof=1) if len(x) > 1 else 0.0
    return mean, sd


def plot_bar(metric_name, values_dh, values_ndh, out_path):
    """Bar plot mean ± SD for DH vs NDH."""
    means = []
    sds = []

    m, sd = mean_sd(values_dh)
    means.append(m)
    sds.append(sd)

    m, sd = mean_sd(values_ndh)
    means.append(m)
    sds.append(sd)

    x = np.arange(2)

    plt.figure(figsize=(4.5, 5))
    plt.bar(x, means, yerr=sds, capsize=6)

    #overlay individual trials
    jitter = 0.08
    plt.scatter(np.full(len(values_dh), x[0]) + np.random.uniform(-jitter, jitter, len(values_dh)),
                values_dh, marker="x")
    plt.scatter(np.full(len(values_ndh), x[1]) + np.random.uniform(-jitter, jitter, len(values_ndh)),
                values_ndh, marker="x")

    plt.xticks(x, ["DH", "NDH"])
    plt.ylabel("Range of Motion (deg)")
    plt.title(f"{metric_name} ROM (mean ± SD)")
    plt.tight_layout()

    plt.savefig(out_path, dpi=200)
    plt.show()
    plt.close()


# =========================
# MAIN
# =========================
def main():
    df = pd.read_excel(XLSX_FILE)

    parsed = df[GROUP_COL].apply(parse_subject_and_hand)
    df["subject_id"] = parsed.apply(lambda x: x[0])
    df["hand"] = parsed.apply(lambda x: x[1])

    for col in THUMB_COLS + INDEX_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # =========================
    # THUMB
    # =========================
    thumb_dh = df[df["hand"] == "DH"][THUMB_COLS].values.flatten()
    thumb_ndh = df[df["hand"] == "NDH"][THUMB_COLS].values.flatten()

    thumb_dh = thumb_dh[np.isfinite(thumb_dh)]
    thumb_ndh = thumb_ndh[np.isfinite(thumb_ndh)]

    plot_bar(
        metric_name="Thumb",
        values_dh=thumb_dh,
        values_ndh=thumb_ndh,
        out_path=BASE_DIR / "thumb_ROM_DH_vs_NDH.png"
    )

    # =========================
    # INDEX
    # =========================
    index_dh = df[df["hand"] == "DH"][INDEX_COLS].values.flatten()
    index_ndh = df[df["hand"] == "NDH"][INDEX_COLS].values.flatten()

    index_dh = index_dh[np.isfinite(index_dh)]
    index_ndh = index_ndh[np.isfinite(index_ndh)]

    plot_bar(
        metric_name="Index",
        values_dh=index_dh,
        values_ndh=index_ndh,
        out_path=BASE_DIR / "index_ROM_DH_vs_NDH.png"
    )

    print("Figures generated in:", BASE_DIR)


if __name__ == "__main__":
    main()
