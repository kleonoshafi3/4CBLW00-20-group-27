import pandas as pd
from pathlib import Path

# --- load and filter as before ---
data_dir = Path(r"C:\Users\20220848\OneDrive - TU Eindhoven\Desktop\TuE\3rd year\Q4\Data Challenge 2\data")

# collect all BTP street files
btp_dfs = []
for month_dir in sorted(data_dir.iterdir()):
    if month_dir.is_dir():
        f = month_dir / f"{month_dir.name}-btp-street.csv"
        if f.exists():
            btp_dfs.append(pd.read_csv(f))
btp_df = pd.concat(btp_dfs, ignore_index=True)

# filter for Burglary in London
mask = (
    (btp_df['Crime type'] == 'Burglary') &
    (btp_df['Location'].str.contains('London', case=False, na=False))
)
london_burglary = btp_df.loc[mask]

# --- print the 15 cases ---
print("=== London Burglary Cases (n=15) ===")
print(london_burglary)

# --- check for any nulls in those rows ---
null_counts = london_burglary.isnull().sum()
print("\n=== Null counts per column ===")
print(null_counts)

# --- (optional) show any rows that have at least one null ---
null_rows = london_burglary[london_burglary.isnull().any(axis=1)]
if null_rows.empty:
    print("\nNo rows contain null values.")
else:
    print("\nRows with at least one null value:")
    print(null_rows)
