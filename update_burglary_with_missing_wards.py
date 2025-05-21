#!/usr/bin/env python3
"""
update_burglary_with_missing_wards.py

1) Reads the current 'burglary_cases_with_ward.csv'.
2) Reads 'missing_lsoas_2011_ward.csv' which maps the previously missing LSOAs to Ward ID/Name.
3) Fills in Ward ID/Name for those rows.
4) Overwrites 'burglary_cases_with_ward.csv' with the fully populated file.
"""

import pandas as pd
from pathlib import Path

def main():
    # Paths
    burglary_csv = Path('output_csv_files') / 'burglary_cases_with_ward.csv'
    missing_map  = Path('output_csv_files') / 'missing_lsoas_2011_ward.csv'

    # 1) Load the existing enriched burglary file
    print(f"Loading burglary data from {burglary_csv}")
    df = pd.read_csv(burglary_csv, dtype=str)
    print(f"  → {len(df):,} rows loaded; missing wards: {df['Ward Name'].isna().sum():,}")

    # 2) Load the mapping for the previously missing LSOAs
    print(f"Loading missing-LSOA→Ward map from {missing_map}")
    map_df = pd.read_csv(missing_map, dtype=str)[['LSOA code','Ward ID','Ward Name']]
    print(f"  → {len(map_df):,} mappings loaded")

    # 3) Merge to bring in the new Ward ID/Name columns
    merged = df.merge(
        map_df,
        on='LSOA code',
        how='left',
        suffixes=('', '_new')
    )

    # 4) Fill missing wards from the new columns
    merged['Ward ID'] = merged['Ward ID'].fillna(merged['Ward ID_new'])
    merged['Ward Name'] = merged['Ward Name'].fillna(merged['Ward Name_new'])

    # 5) Drop the helper columns
    merged = merged.drop(columns=['Ward ID_new', 'Ward Name_new'])

    # 6) Confirm no more missing wards
    missing_after = merged['Ward Name'].isna().sum()
    print(f"Missing wards after update: {missing_after:,}")

    # 7) Overwrite the original file with the complete data
    merged.to_csv(burglary_csv, index=False)
    print(f"All done — '{burglary_csv.name}' has been updated with complete ward assignments.")

if __name__ == '__main__':
    main()
