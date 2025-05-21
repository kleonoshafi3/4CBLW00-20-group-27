#!/usr/bin/env python3
"""
map_missing_lsoas_2011.py

1) Reads 'output_csv_files/missing_lsoas.csv' (unique LSOA codes missing wards).
2) Reads the 2011 LSOA→Ward lookup from 'data/switching/lsoa_to_ward_2011.xlsx'.
3) Renames columns so we can merge on the same key.
4) Merges and writes 'output_csv_files/missing_lsoas_2011_ward.csv'.
"""

import pandas as pd
from pathlib import Path

def main():
    # Paths
    missing_csv    = Path('output_csv_files') / 'missing_lsoas.csv'
    lookup_xlsx    = Path('data/switching') / 'lsoa_to_ward_2011.xlsx'
    output_csv     = Path('output_csv_files') / 'missing_lsoas_2011_ward.csv'

    # 1) Load list of missing LSOAs
    print(f"Loading missing LSOAs from {missing_csv}")
    miss_df = pd.read_csv(missing_csv, dtype=str)
    print(f"  → {len(miss_df):,} missing LSOAs")

    # 2) Load the 2011 LSOA→Ward lookup Excel
    print(f"Loading 2011 lookup from {lookup_xlsx}")
    lookup_df = pd.read_excel(lookup_xlsx, dtype=str)
    print(f"  → {len(lookup_df):,} lookup rows before renaming")

    # 3) Rename columns to match
    lookup_df = lookup_df.rename(columns={
        'LSOA11CD': 'LSOA code',   # old LSOA key
        'WD21CD':   'Ward ID',
        'WD21NM':   'Ward Name'
    })
    # If your excel uses WD24CD / WD24NM instead, swap those names above.

    # 4) Merge
    print("Merging missing LSOAs with 2011-ward lookup...")
    merged = miss_df.merge(lookup_df[['LSOA code','Ward ID','Ward Name']],
                            on='LSOA code', how='left')
    still_missing = merged['Ward Name'].isna().sum()
    print(f"  → {still_missing:,} of the missing LSOAs still have no ward")

    # 5) Save
    print(f"Saving results to {output_csv}")
    merged.to_csv(output_csv, index=False)
    print("Done.")

if __name__ == '__main__':
    main()
