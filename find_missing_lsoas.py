#!/usr/bin/env python3
"""
find_missing_lsoas.py

Reads 'burglary_cases_with_ward.csv' and prints (and saves) the list of LSOA codes
that still have no ward assignment.
"""

import pandas as pd
from pathlib import Path

def find_missing_lsoas():
    # 1) Load the enriched burglary file
    fn = Path('output_csv_files') / 'burglary_cases_with_ward.csv'
    df = pd.read_csv(fn, dtype=str)

    # 2) Identify rows where Ward Name is missing or blank
    missing_mask = df['Ward Name'].isna() | (df['Ward Name'].str.strip() == '')
    missing = df.loc[missing_mask, 'LSOA code']

    # 3) Count occurrences per LSOA
    counts = missing.value_counts().rename_axis('LSOA code').reset_index(name='Missing Count')

    # 4) Print results
    if counts.empty:
        print("All LSOAs have a ward assigned.")
    else:
        print(f" {len(counts):,} unique LSOA codes missing a ward:")
        print(counts.to_string(index=False, max_rows=20))
        print(f"\nTotal missing rows: {missing_mask.sum():,}")

        # 5) Save to CSV for further inspection
        out_csv = Path('output_csv_files') / 'missing_lsoas.csv'
        counts.to_csv(out_csv, index=False)
        print(f"Saved full list to: {out_csv}")

if __name__ == '__main__':
    find_missing_lsoas()
