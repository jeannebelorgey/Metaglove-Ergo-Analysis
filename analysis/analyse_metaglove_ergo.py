import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ---------- LOADING ----------

def load_metaglove_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a MetaGlove CSV file and create a time_s column in seconds.

    It looks for one of:
    - Elapsed_Time_In_Milliseconds
    - Time
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    time_col = None
    for c in ["Elapsed_Time_In_Milliseconds", "Time", "time"]:
        if c in df.columns:
            time_col = c
            break

    if time_col is None:
        raise ValueError("No time column found (Elapsed_Time_In_Milliseconds / Time).")

    # convert to seconds
    if "Millisecond" in time_col:
        df["time_s"] = df[time_col] / 1000.0
    else:
        df["time_s"] = df[time_col].astype(float)

    return df


# ---------- HELPER ----------

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
JOINTS = ["CMC", "MCP", "PIP", "DIP", "TIP"]


def has_columns(df: pd.DataFrame, cols: list[str]) -> bool:
    """Return True if ALL columns are present in the dataframe."""
    return all(c in df.columns for c in cols)


# ---------- PLOTS ----------

def plot_hand_trajectory(df: pd.DataFrame, title_suffix: str = ""):
    """
    Plot the hand trajectory in the X-Y plane and Y-Z plane.

    Uses Hand_Position_X / Y / Z.
    """
    pos_cols = ["Hand_Position_X", "Hand_Position_Y", "Hand_Position_Z"]
    if not has_columns(df, pos_cols):
        print("[INFO] No Hand_Position_X/Y/Z columns, cannot plot hand trajectory.")
        return

    # X-Y projection
    plt.figure()
    plt.plot(df["Hand_Position_X"], df["Hand_Position_Y"])
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title(f"Hand trajectory (X-Y){title_suffix}")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    # Y-Z projection
    plt.figure()
    plt.plot(df["Hand_Position_Y"], df["Hand_Position_Z"])
    plt.xlabel("Y position")
    plt.ylabel("Z position")
    plt.title(f"Hand trajectory (Y-Z){title_suffix}")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_finger_joint_flexion(df: pd.DataFrame, finger: str):
    """
    Plot flexion angles over time for all joints of a given finger
    (MCP, PIP, DIP, and CMC if available).

    Uses columns like:
    - Index_MCP_Flex
    - Index_PIP_Flex
    - Index_DIP_Flex
    """
    finger = finger.capitalize()
    cols = []
    for joint in JOINTS:
        col = f"{finger}_{joint}_Flex"
        if col in df.columns:
            cols.append((joint, col))

    if not cols:
        print(f"[INFO] No flexion columns found for finger {finger}.")
        return

    plt.figure()
    for joint, col in cols:
        plt.plot(df["time_s"], df[col], label=f"{joint}_Flex")

    plt.xlabel("Time (s)")
    plt.ylabel("Flexion angle (°)")
    plt.title(f"{finger} finger - joint flexion over time")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_joint_all_fingers(df: pd.DataFrame, joint: str = "MCP", angle_type: str = "Flex"):
    """
    Plot the same joint (e.g. MCP Flex) for all fingers on one graph.

    Example:
    - Index_MCP_Flex
    - Middle_MCP_Flex
    - Ring_MCP_Flex
    - Pinky_MCP_Flex
    """
    joint = joint.upper()
    angle_type = angle_type.capitalize()

    cols = []
    for finger in FINGERS:
        col = f"{finger}_{joint}_{angle_type}"
        if col in df.columns:
            cols.append((finger, col))

    if not cols:
        print(f"[INFO] No columns found for joint {joint} and angle type {angle_type}.")
        return

    plt.figure()
    for finger, col in cols:
        plt.plot(df["time_s"], df[col], label=finger)

    plt.xlabel("Time (s)")
    plt.ylabel(f"{joint} {angle_type} angle (°)")
    plt.title(f"{joint} {angle_type} for all fingers")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pinch_distances(df: pd.DataFrame):
    """
    Plot pinch distances between thumb and each finger over time.

    Uses:
    - Pinch_ThumbToIndex
    - Pinch_ThumbToMiddle
    - Pinch_ThumbToRing
    - Pinch_ThumbToPinky
    """
    pinch_cols = [
        "Pinch_ThumbToIndex",
        "Pinch_ThumbToMiddle",
        "Pinch_ThumbToRing",
        "Pinch_ThumbToPinky",
    ]
    available = [c for c in pinch_cols if c in df.columns]

    if not available:
        print("[INFO] No pinch distance columns found.")
        return

    plt.figure()
    for col in available:
        plt.plot(df["time_s"], df[col], label=col.replace("Pinch_", "").replace("ThumbTo", "Thumb-"))

    plt.xlabel("Time (s)")
    plt.ylabel("Pinch distance (units from Manus)")
    plt.title("Thumb–finger pinch distances over time")
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_mcp_rom(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MCP flexion range of motion (ROM = max - min) for each finger.

    Returns a dataframe with columns:
    - finger
    - ROM_deg
    """
    records = []
    for finger in FINGERS:
        col = f"{finger}_MCP_Flex"
        if col not in df.columns:
            continue
        values = df[col].dropna()
        if values.empty:
            continue
        rom = values.max() - values.min()
        records.append({"finger": finger, "ROM_deg": rom})

    return pd.DataFrame(records)


def plot_mcp_rom_bar(df: pd.DataFrame):
    """
    Bar plot of MCP flexion ROM for all fingers.

    This gives a quick overview of which finger has the smallest / largest
    MCP movement during the task.
    """
    rom_df = compute_mcp_rom(df)
    if rom_df.empty:
        print("[INFO] No MCP_Flex columns found, cannot compute ROM.")
        return

    plt.figure()
    plt.bar(rom_df["finger"], rom_df["ROM_deg"])
    plt.xlabel("Finger")
    plt.ylabel("MCP flexion ROM (°)")
    plt.title("MCP flexion range of motion per finger")
    plt.tight_layout()
    plt.show()


# ---------- MAIN ANALYSIS FUNCTION ----------

def analyze_session(csv_path: str | Path, label: str = ""):
    """
    High-level analysis of one MetaGlove recording.

    It produces:
    - hand trajectory plots
    - flexion curves for each joint of each finger (called manually)
    - MCP flexion for all fingers on one plot
    - thumb–finger pinch distance curves
    - MCP ROM bar plot
    """
    print(f"\n===== Analyzing session: {csv_path} ({label}) =====")

    df = load_metaglove_csv(csv_path)

    # 1) Hand trajectory
    plot_hand_trajectory(df, title_suffix=f" - {label}")

    # 2) Flexion of all joints for each finger (you can comment some out if too many plots)
    for finger in FINGERS:
        plot_finger_joint_flexion(df, finger)

    # 3) Same joint for all fingers at once (MCP flexion is very relevant in ergotherapy)
    plot_joint_all_fingers(df, joint="MCP", angle_type="Flex")

    # 4) Pinch distances (thumb–index, thumb–middle, etc.)
    plot_pinch_distances(df)

    # 5) MCP ROM per finger
    plot_mcp_rom_bar(df)


# ---------- ENTRY POINT ----------

if __name__ == "__main__":
    # Change this path to the CSV you want to analyze
    # Example for your right-hand file:
    csv_file = Path(r"C:\Users\jeann\Documents\MetaGlove_Project\data\drink from a flask both hands_CIIRC_R.csv")

    if csv_file.exists():
        analyze_session(csv_file, label="Right hand - drink from a flask")
    else:
        print("CSV file not found, please check the path:", csv_file)