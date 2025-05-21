import pandas as pd

# Load the dataset
df = pd.read_csv('output_csv_files/burglary_cases_with_ward_cleaned.csv')

# Ensure 'Location' column exists
if 'Location' not in df.columns:
    print("The 'Location' column is missing.")
else:
    # Get unique location values (excluding nulls)
    unique_locations = df['Location'].dropna().unique()
    unique_locations = sorted(unique_locations)

    # Write to CSV
    out_df = pd.DataFrame({'Unique_Location': unique_locations})
    out_df.to_csv('output_csv_files/unique_locations.csv', index=False)

    print("Unique locations written to: output_csv_files/unique_locations.csv")
