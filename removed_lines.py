import pandas as pd

# Load both datasets
original_df = pd.read_csv('output_csv_files/burglary_cases_with_ward_cleaned.csv')
residential_df = pd.read_csv('output_csv_files/residential_burglaries.csv')

# Use Crime ID to identify removed entries (assuming Crime ID is unique and reliable)
removed_df = original_df[~original_df['Crime ID'].isin(residential_df['Crime ID'])].copy()

# Output the removed rows
removed_df.to_csv('output_csv_files/filtered_out_non_residential.csv', index=False)
print(f"{len(removed_df)} rows removed. Saved to output_csv_files/filtered_out_non_residential.csv")
