import pandas as pd
from pathlib import Path

def main():
    input_path = Path('output_csv_files') / 'burglary_cases_with_ward.csv'
    output_counts_path = Path('output_csv_files') / 'ward_burglary_counts.csv'
    output_stats_path = Path('output_csv_files') / 'ward_burglary_stats.csv'

    # Load the enriched burglary data
    df = pd.read_csv(input_path, dtype=str)

    # Count burglaries per ward
    counts = (
        df['Ward Name']
        .fillna('Unknown')
        .value_counts()
        .rename_axis('Ward Name')
        .reset_index(name='Burglary Count')
    )

    # Save ward-level burglary counts
    counts.to_csv(output_counts_path, index=False)

    # Compute descriptive statistics over the counts
    stats = counts['Burglary Count'].describe()

    # Convert stats to a DataFrame
    stats_df = stats.rename_axis('Statistic').reset_index()
    stats_df.columns = ['Statistic', 'Value']

    # Save the summary statistics to CSV
    stats_df.to_csv(output_stats_path, index=False)

    print(f"Wrote counts to: {output_counts_path}")
    print(f"Wrote stats  to: {output_stats_path}")

if __name__ == "__main__":
    main()
