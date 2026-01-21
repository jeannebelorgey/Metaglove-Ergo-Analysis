# MetaGlove + Box and Block Test (BBT): Pinch Event Detection & ROM Metrics

This repository contains Python scripts to analyze Manus MetaGlove motion capture exports (CSV) recorded during the Box and Block Test (BBT). The pipeline focuses on:
- Thumb–index pinch signal (`Pinch_ThumbToIndex`) for grasp event detection 
- Range of motion (ROM) metrics for thumb CMC flexion and index MCP flexion
- Summary outputs (PNG figure + CSV summary) and ROM comparison barplots (Dominant Hand vs Non Domniant Hand)

## Context
The classical BBT provides a global performance score (number of blocks transferred in 60 s) but does not directly describe movement quality. Instrumented variants and wearable sensing approaches motivate extracting kinematic indicators beyond the final score. [1][2][3][4][5][6]

## Data format 
This code expects Manus Core CSV exports to include:
- A time column: `Elapsed_Time_In_Milliseconds` or `Time` or `Frame`
- Pinch signal: `Pinch_ThumbToIndex`
- Joint angles (degrees): `Thumb_CMC_Flex`, `Index_MCP_Flex`

## Scripts
### 1) `src/analyze_metaglove_ergo.py`
Input: one Manus CSV file  
Output:
- `*_pinch_events.png`: pinch signal + detected close/open events + thresholds
- `*_summary.csv`: number of grasps, mean/median grasp duration, ROM metrics (P95–P5) [7], thresholds

### 2) `src/rom_barplots.py`
Input: an Excel file (`ROM_data.xlsx`) with ROM values across trials (DH and NDH)  
Output: barplots with trial points for:
- Thumb ROM (Dominant Hand vs Non-Dominant Hand)
- Index ROM (Domninant Hand vs Non-Dominant Hand)
  
## Repository structure

The repository can be organized like:

```
MetaGlove_Project/
│
├── analysis/
│ ├── analyse_metaglove_ergo.py
│
├── data/
│ ├── example_BBT.csv
│
├── output/
│
├── Barplot/
│ ├── rom_barplots.py
│ ├── ROM_data.xlsx
│
├── README.md
```

## Requirements

The scripts require Python 3 and the following libraries:

- numpy  
- pandas  
- matplotlib
- openpyxl  

## Running analyse_metaglove_ergo.py

The Metaglove CSV files can be placed in the data/ folder
Edit the paths in the ```__main__``` section: 
```
csv_file = Path(r"C:\...\MetaGlove_Project\data\BBTfile.csv")
out_dir  = Path(r"C:\...\MetaGlove_Project\output")
```
Then run: 
```
python analyse_metaglove_ergo.py
```
## Running rom_barplots.py

The Excel file containing ROM values (e.g., ROM_data.xlsx) can be placed in the Boxplot/ folder
Edit the paths at the beginning of the script:
```
BASE_DIR = Path(r"C:\...\MetaGlove_Project\Barplot")
XLSX_FILE = BASE_DIR / "ROM_data.xlsx"
```

Then run: 
```
python rom_barplots.py
```

## References
[1] Lee, S., Lee, H., Lee, J., Ryu, H., Kim, I. Y., & Kim, J. (2020). Clip-On IMU System for Assessing Age-Related Changes in Hand Functions. Sensors, 20(21), 6313. https://doi.org/10.3390/s20216313
[2] Cocco, E. S., Pournajaf, S., Romano, P., Morone, G., Thouant, C.-L., Buscarini, L., Manzia, C. M., Cioeta, M., Felzani, G., Infarinato, F., Franceschini, M., & Goffredo, M. (2024). Comparative analysis of upper body kinematics in stroke, Parkinson’s disease, and healthy subjects : An observational study using IMU-based targeted box and block test. Gait & Posture, 114, 69‑77. https://doi.org/10.1016/j.gaitpost.2024.09.002
[3] Lemos, J. D., Hernandez, A. M., & Soto-Romero, G. (2017). An Instrumented Glove to Assess Manual Dexterity in Simulation-Based Neurosurgical Education. Sensors, 17(5), 988. https://doi.org/10.3390/s17050988
[4] Vanmechelen, I., Haberfehlner, H., De Vleeschhauwer, J., Van Wonterghem, E., Feys, H., Desloovere, K., Aerts, J.-M., & Monbaliu, E. (2023). Assessment of movement disorders using wearable sensors during upper limb tasks : A scoping review. Frontiers in Robotics and AI, 9, 1068413. https://doi.org/10.3389/frobt.2022.1068413 
[5] Saggio, G., Roselli, P., Pietrosanti, L., Romano, A., Arangino, N., Patera, M., & Suppa, A. (2025). A New Geometric Algebra-Based Classification of Hand Bradykinesia in Parkinson’s Disease Measured Using a Sensory Glove. Algorithms, 18(8), 527. https://doi.org/10.3390/a18080527
[6] Quasi-Static and Dynamic Measurement Capabilities Provided by an Electromagnetic Field-Based Sensory Glove. (s. d.). https://www.mdpi.com/2079-6374/15/10/640
[7] Janice M. Moreside, Stuart M. McGill, Quantifying normal 3D hip ROM in healthy young adult males with clinical and laboratory tools: Hip mobility restrictions appear to be plane-specific,Clinical Biomechanics,Volume 26, Issue 8,2011,Pages 824-829,ISSN 0268-0033, https://www.sciencedirect.com/science/article/abs/pii/S0268003311000982 
