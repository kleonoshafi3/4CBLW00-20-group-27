"""
Data Loader for Crime Data Analysis
This script loads crime data from CSV files and creates separate CSV files for different crime categories.
"""

import pandas as pd
from pathlib import Path
import sqlite3
import os

def load_all_data():
    """Load and combine all monthly data from the data directory and create a dataframe"""
    print("Loading data from CSV files...")
    data_dir = Path('data')
    all_data = []
    
    for month_dir in sorted(data_dir.iterdir()):
        if month_dir.is_dir():
            try:
                # Load both metropolitan and city of london data
                met_street = pd.read_csv(month_dir / f"{month_dir.name}-metropolitan-street.csv")
                city_street = pd.read_csv(month_dir / f"{month_dir.name}-city-of-london-street.csv")
                combined = pd.concat([met_street, city_street], ignore_index=True)
                all_data.append(combined)
                print(f"Loaded data from {month_dir.name}")
            except Exception as e:
                print(f"Error loading {month_dir.name}: {e}")
    
    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal records loaded: {len(df)}")
    return df

def perform_eda(df):
    """Explore the data and check missing values"""
    print("\n=== Data Exploration ===")
    
    # 1. Check initial data shape and info
    print("\nInitial Data Shape:", df.shape)
    print("\nInitial Data Info:")
    print(df.info())
    print(df.describe())
    
    # 2. Handle missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # 3. Crime Type Analysis
    print("\n=== Crime Type Distribution ===")
    crime_types = df['Crime type'].value_counts()
    print(crime_types)
    
    # Create SQLite database and table
    conn = sqlite3.connect(':memory:')
    df.to_sql('crime_data', conn, index=False)
    
    # Check missing values in Last outcome category OR Crime ID
    print("\n=== Missing Values Analysis ===")
    
    # Count missing values in each column
    missing_outcome_query = """
    SELECT COUNT(*) as count FROM crime_data 
    WHERE "Last outcome category" IS NULL
    """
    missing_id_query = """
    SELECT COUNT(*) as count FROM crime_data 
    WHERE "Crime ID" IS NULL
    """
    missing_antisocial_query = """
    SELECT COUNT(*) as count FROM crime_data 
    WHERE "Crime type" = 'Anti-social behaviour' 
    AND ("Last outcome category" IS NULL OR "Crime ID" IS NULL)
    """
    
    missing_outcome_count = pd.read_sql_query(missing_outcome_query, conn)['count'][0]
    missing_id_count = pd.read_sql_query(missing_id_query, conn)['count'][0]
    missing_antisocial_count = pd.read_sql_query(missing_antisocial_query, conn)['count'][0]
    
    print(f"Number of missing Last outcome category: {missing_outcome_count}")
    print(f"Number of missing Crime ID: {missing_id_count}")
    print(f"Number of missing values in anti-social behaviour cases: {missing_antisocial_count}")
    
    # Check if the same rows have missing values in all three categories
    same_missing_query = """
    SELECT COUNT(*) as count FROM crime_data 
    WHERE "Last outcome category" IS NULL 
    AND "Crime ID" IS NULL
    AND "Crime type" = 'Anti-social behaviour'
    """
    same_missing_count = pd.read_sql_query(same_missing_query, conn)['count'][0]
    
    print(f"\nNumber of rows with all three missing: {same_missing_count}")
    
    # Check pairwise overlaps
    print("\n=== Pairwise Missing Value Overlaps ===")
    
    # Last outcome category and Crime ID
    outcome_id_overlap = """
    SELECT COUNT(*) as count FROM crime_data 
    WHERE "Last outcome category" IS NULL AND "Crime ID" IS NULL
    """
    outcome_id_count = pd.read_sql_query(outcome_id_overlap, conn)['count'][0]
    print(f"Rows with missing Last outcome category AND Crime ID: {outcome_id_count}")
    
    # Last outcome category and Anti-social behaviour
    outcome_antisocial_overlap = """
    SELECT COUNT(*) as count FROM crime_data 
    WHERE "Last outcome category" IS NULL AND "Crime type" = 'Anti-social behaviour'
    """
    outcome_antisocial_count = pd.read_sql_query(outcome_antisocial_overlap, conn)['count'][0]
    print(f"Rows with missing Last outcome category AND Anti-social behaviour: {outcome_antisocial_count}")
    
    # Crime ID and Anti-social behaviour
    id_antisocial_overlap = """
    SELECT COUNT(*) as count FROM crime_data 
    WHERE "Crime ID" IS NULL AND "Crime type" = 'Anti-social behaviour'
    """
    id_antisocial_count = pd.read_sql_query(id_antisocial_overlap, conn)['count'][0]
    print(f"Rows with missing Crime ID AND Anti-social behaviour: {id_antisocial_count}")
    
    # Calculate percentage of overlap
    if missing_outcome_count > 0 and missing_id_count > 0:
        overlap_percentage = (same_missing_count / max(missing_outcome_count, missing_id_count, missing_antisocial_count)) * 100
        print(f"\nPercentage of missing values that overlap in all three: {overlap_percentage:.2f}%")
    
    # Verify if all counts are the same
    if missing_outcome_count == missing_id_count == missing_antisocial_count:
        print("\nAll missing value counts are equal!")
        if missing_outcome_count == 710193:
            print("And they all match the expected count of 710193")
        else:
            print("But they don't match the expected count of 710193")
    else:
        print("\nWarning: The missing value counts are not equal")
        if missing_outcome_count == 710193 or missing_id_count == 710193 or missing_antisocial_count == 710193:
            print("At least one count matches the expected count of 710193")
    
    # Specific verification checks
    print("\n=== Specific Verification Checks ===")
    
    # Check if missing Crime ID implies missing Last outcome and Anti-social
    if missing_id_count == same_missing_count:
        print("ALL rows with missing Crime ID are anti-social and have missing outcome too")
    else:
        print("NOT ALL rows with missing Crime ID are anti-social and missing outcome — some have other values")
    
    # Check if missing Last outcome implies missing Crime ID and Anti-social
    if missing_outcome_count == same_missing_count:
        print("ALL rows with missing Last outcome are anti-social and have missing Crime ID too")
    else:
        print("NOT ALL rows with missing Last outcome are anti-social and missing Crime ID — some have other values")
    
    # Check if Anti-social implies missing Last outcome and Crime ID
    if missing_antisocial_count == same_missing_count:
        print("ALL anti-social cases have both Last outcome and Crime ID missing")
    else:
        print("NOT ALL anti-social cases have both Last outcome and Crime ID missing — some have values")
    
    # Close the database connection
    conn.close()
    
    return df

def create_separate_csv_files(df):
    """Create separate CSV files"""
    print("\n=== Creating Separate CSV Files ===")
    
    # Create output directory if it doesn't exist
    output_dir = Path('output_csv_files')
    output_dir.mkdir(exist_ok=True)
    
    # Create SQLite database and table
    conn = sqlite3.connect(':memory:')
    df.to_sql('crime_data', conn, index=False)
    
    # 1. Burglary cases
    print("\nCreating burglary cases file...")
    burglary_query = """
    SELECT * FROM crime_data 
    WHERE "Crime type" = 'Burglary'
    """
    burglary_df = pd.read_sql_query(burglary_query, conn)
    burglary_df.to_csv(output_dir / 'burglary_cases.csv', index=False)
    print(f"Created burglary_cases.csv with {len(burglary_df)} records")
    
    # Close the database connection
    conn.close()
    
    print("\nProcess completed!")

def preprocess_data(df):
    """Preprocess the data by removing unnecessary columns and handling missing values"""
    print("\n=== Data Preprocessing ===")
    
    # Print initial columns and check for context
    print("\nInitial columns:")
    print(df.columns.tolist())
    
    # Check for context column (case-insensitive)
    context_col = None
    for col in df.columns:
        if col.lower() == 'context':
            context_col = col
            break
    
    print("\nChecking for 'context' column before removal:")
    print("'context' column exists:", context_col is not None)
    if context_col:
        print(f"Found column: '{context_col}'")
    
    # Remove the context column if found
    if context_col:
        df = df.drop(columns=[context_col])
        print(f"\nRemoved '{context_col}' column as all values are missing")
    
    # Verify the removal
    print("\nVerifying 'context' column after removal:")
    remaining_context = any(col.lower() == 'context' for col in df.columns)
    print("'context' column exists:", remaining_context)
    
    # Print the shape after preprocessing
    print(f"\nData shape after preprocessing: {df.shape}")
    print("\nColumns after preprocessing:")
    print(df.columns.tolist())
    
    return df

def main():
    """Main function to load data and create separate CSV files"""
    # Load all data
    df = load_all_data()
    # Perform EDA and check missing values
    df = perform_eda(df)
    # Preprocess the data
    df = preprocess_data(df)
    # Create separate CSV files
    create_separate_csv_files(df)
    
    print("\nData loading and analysis completed successfully!")

if __name__ == "__main__":
    main()
