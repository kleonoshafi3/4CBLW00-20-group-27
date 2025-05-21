#!/usr/bin/env python3
"""
clean_burglary_no_location.py

Removes rows from 'burglary_cases_with_ward.csv' where:
- Location is "No Location" OR
- LSOA code is missing or empty

Saves the result to 'burglary_cases_with_ward_cleaned.csv'
"""

import pandas as pd
from pathlib import Path

def main():
    input_file = Path('output_csv_files') / 'burglary_cases_with_ward.csv'
    output_file = Path('output_csv_files') / 'burglary_cases_with_ward_cleaned.csv'

    # Load CSV
    print(f"Loading data from {input_file}")
    df = pd.read_csv(input_file, dtype=str)
    print(f"  → Loaded {len(df):,} rows")

    # Drop rows with "No Location" or missing/blank LSOA
    mask = (df['Location'] != 'No Location') & df['LSOA code'].notna() & (df['LSOA code'].str.strip() != '')
    cleaned_df = df[mask]

    print(f"  → Remaining after cleaning: {len(cleaned_df):,} rows")
    print(f"  → Removed: {len(df) - len(cleaned_df):,} rows")

    # Save cleaned version
    cleaned_df.to_csv(output_file, index=False)
    print(f"Cleaned file saved to: {output_file}")

if __name__ == '__main__':
    main()
