#!/usr/bin/env python3
"""
count_wards_from_lsoa.py

Generates a CSV counting how many unique LSOAs fall under each ward,
based on the mapping in 'data/lsoa_to_ward.csv'.
"""

import pandas as pd
from pathlib import Path

def main():
    # Input and output paths
    input_file = Path('data') / 'switching/lsoa_to_ward.csv'
    output_file = Path('output_csv_files') / 'lsoa_count_per_ward.csv'

    # Load the LSOA to Ward mapping
    print(f"Loading LSOA to Ward mapping from {input_file}")
    df = pd.read_csv(input_file, dtype=str)

    # Drop rows with missing ward name
    df = df.dropna(subset=['WD24NM'])

    # Count how many LSOAs per ward
    print("Counting LSOAs per ward...")
    ward_counts = df.groupby(['WD24CD', 'WD24NM']).size().reset_index(name='LSOA Count')

    # Sort by LSOA count descending
    ward_counts = ward_counts.sort_values(by='LSOA Count', ascending=False)

    # Save to CSV
    ward_counts.to_csv(output_file, index=False)
    print(f"Saved ward counts to: {output_file}")

if __name__ == '__main__':
    main()
