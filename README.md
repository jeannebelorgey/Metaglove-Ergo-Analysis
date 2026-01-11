# Metaglove CSV analysis for NHPT / BBT (Manus / Xsens)

This project analyzes motion capture CSV data recorded with Manus/Xsens Metagloves during:
- NHPT (Nine-Hole Peg Test)
- BBT (Box and Block Test)

It detects pinch close/open events (thumb-to-index pinch), computes grasp-related metrics, and exports plots and per-file summaries.

## Features

Given a Metaglove CSV file, the script:
- Builds a time vector in seconds (`t_s`) and estimates sampling rate (`fs_hz`)
- Infers test type from the filename: `NHPT` / `BBT`
- Detects pinch close/open events with hysteresis thresholds 
- Computes metrics:
  - number of grasps
  - grasp durations (close → open)
  - cycle durations (close → close) + variability (CV)
  - inter-grasp intervals (open → next close)
  - joint angle ROM (P95 − P5)
- Saves 5 plots and a summary CSV per file
  
## Repository structure

The repository can be organized like:

```
MetaGlove_Project/
│
├── analysis/
│ ├── analyse_metaglove_ergo.py
│
├── data/
│ ├── example_NHPT.csv
│ ├── example_BBT.csv
│
├── output/
│
├── README.md
```

## Requirements

The script requires Python 3 and the following libraries:

- numpy  
- pandas  
- matplotlib  

They can be installed using:

```bash
pip install numpy pandas matplotlib
```
## Running the analysis

The Metaglove CSV files can be placed in the data/ folder
Edit the paths in the __main__ section: 
csv_file = Path(r"C:\...\MetaGlove_Project\data\BBT or NHPT file.csv")
out_dir  = Path(r"C:\...\MetaGlove_Project\output")
Then run: 
python analyse_metaglove_ergo.py
