import pandas as pd
from pathlib import Path

def main():
    # Paths
    lookup_path = Path('data') / 'lsoa_to_ward.csv'
    burglary_path = Path('output_csv_files') / 'burglary_cases.csv'
    output_path = Path('output_csv_files') / 'burglary_cases_with_ward.csv'
    
    # 1) Load lookup table, picking the relevant columns
    lookup = pd.read_csv(
        lookup_path,
        usecols=['LSOA21CD', 'WD24CD', 'WD24NM'],
        dtype=str
    ).rename(columns={
        'LSOA21CD': 'LSOA code',
        'WD24CD': 'Ward ID',
        'WD24NM': 'Ward Name'
    })
    
    # 2) Load burglary data
    df = pd.read_csv(burglary_path, dtype=str)
    
    # 3) Merge on LSOA code
    merged = df.merge(lookup, on='LSOA code', how='left')
    
    # 4) Save out the enriched file
    merged.to_csv(output_path, index=False)
    print(f"Saved enriched burglary file with {len(merged)} rows to {output_path}")

if __name__ == "__main__":
    main()
