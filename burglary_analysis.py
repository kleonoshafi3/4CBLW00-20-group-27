"""
Burglary Analysis Script
This script performs analysis of police demand and effectiveness metrics for burglary prevention in London.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

def load_burglary_data():
    """Load the burglary cases data from CSV"""
    print("Loading burglary cases data...")
    df = pd.read_csv('output_csv_files/burglary_cases.csv')
    print(f"Loaded {len(df)} burglary cases")
    return df

def perform_eda(df):
    """Perform exploratory data analysis on burglary cases"""
    print("\n=== Exploratory Data Analysis ===")
    
    # Basic information
    print("\n1. Basic Information:")
    print(f"Number of records: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumns:")
    print(df.columns.tolist())
    
    # Data types and missing values
    print("\n2. Data Types and Missing Values:")
    print(df.info())
    
    # Missing values analysis
    print("\n3. Missing Values Analysis:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    # Basic statistics
    print("\n4. Basic Statistics:")
    print(df.describe())
    
    # Crime type distribution
    print("\n5. Crime Type Distribution:")
    print(df['Crime type'].value_counts())
    
    # Last outcome category analysis
    print("\n6. Last Outcome Category Analysis:")
    print(df['Last outcome category'].value_counts())
    
    # Temporal analysis
    print("\n7. Temporal Analysis:")
    df['Month'] = pd.to_datetime(df['Month'])
    monthly_counts = df['Month'].dt.to_period('M').value_counts().sort_index()
    print("\nMonthly Burglary Counts:")
    print(monthly_counts)
    
    return df

def preprocess_data(df):
    """Preprocess the burglary data"""
    print("\n=== Data Preprocessing ===")
    
    # Convert Month to datetime
    df['Month'] = pd.to_datetime(df['Month'])
    
    # Extract temporal features
    df['Year'] = df['Month'].dt.year
    df['Month_num'] = df['Month'].dt.month
    
    # Handle missing values
    print("\nHandling missing values...")
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            print(f"Column '{col}' has {missing_count} missing values")
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna('Unknown', inplace=True)
    
    # Print final shape and columns
    print(f"\nFinal data shape: {df.shape}")
    print("\nFinal columns:")
    print(df.columns.tolist())
    
    return df

def calculate_effectiveness_metrics(df):
    """Calculate metrics for measuring police demand and effectiveness"""
    print("\n=== Calculating Effectiveness Metrics ===")
    
    # 1. Response Time Analysis (if available)
    if 'Reported by' in df.columns and 'Month' in df.columns:
        response_times = df.groupby('Reported by')['Month'].count()
        print("\nBurglary Reports per Police Force:")
        print(response_times)
    
    # 2. Clearance Rate Analysis
    clearance_rate = (df['Last outcome category'] != 'Unknown').mean() * 100
    print(f"\nOverall Clearance Rate: {clearance_rate:.2f}%")
    
    # 3. Last Outcome Category Distribution
    print("\nLast Outcome Category Distribution:")
    outcome_counts = df['Last outcome category'].value_counts()
    print(outcome_counts)
    
    # 4. Geographic Hotspot Analysis
    if 'LSOA code' in df.columns:
        hotspot_analysis = df.groupby('LSOA code').size().sort_values(ascending=False)
        print("\nTop 10 High-Demand Areas:")
        print(hotspot_analysis.head(10))
    
    # 5. Temporal Patterns
    print("\nMonthly Patterns:")
    monthly_patterns = df.groupby(['Year', 'Month_num']).size()
    print(monthly_patterns)
    
    return df

def create_effectiveness_visualizations(df):
    """Create visualizations for police demand and effectiveness analysis"""
    print("\n=== Creating Effectiveness Visualizations ===")
    
    # Create output directory for visualizations
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Monthly Burglary Trends with Clearance Rates
    plt.figure(figsize=(15, 8))
    monthly_data = df.groupby('Month').agg({
        'Crime ID': 'count',
        'Last outcome category': lambda x: (x != 'Unknown').mean() * 100
    }).reset_index()
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    ax1.plot(monthly_data['Month'], monthly_data['Crime ID'], 'b-', label='Burglary Count')
    ax2.plot(monthly_data['Month'], monthly_data['Last outcome category'], 'r-', label='Clearance Rate (%)')
    
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Number of Burglaries', color='b')
    ax2.set_ylabel('Clearance Rate (%)', color='r')
    plt.title('Monthly Burglary Trends and Clearance Rates')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'monthly_trends_with_clearance.png')
    plt.close()
    
    # 2. Geographic Hotspot Map
    if 'LSOA code' in df.columns:
        plt.figure(figsize=(12, 8))
        hotspot_data = df.groupby('LSOA code').size().sort_values(ascending=False).head(20)
        hotspot_data.plot(kind='bar')
        plt.title('Top 20 High-Demand Areas (by LSOA)')
        plt.xlabel('LSOA Code')
        plt.ylabel('Number of Burglaries')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'geographic_hotspots.png')
        plt.close()
    
    # 3. Outcome Category Analysis
    plt.figure(figsize=(12, 6))
    outcome_data = df['Last outcome category'].value_counts().head(10)
    outcome_data.plot(kind='bar')
    plt.title('Top 10 Outcome Categories')
    plt.xlabel('Outcome Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'outcome_analysis.png')
    plt.close()
    
    # 4. Temporal Patterns (Heatmap)
    if 'Year' in df.columns and 'Month_num' in df.columns:
        plt.figure(figsize=(12, 8))
        temporal_data = df.groupby(['Year', 'Month_num']).size().unstack()
        sns.heatmap(temporal_data, cmap='YlOrRd', annot=True, fmt='.0f')
        plt.title('Temporal Patterns of Burglaries')
        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.tight_layout()
        plt.savefig(output_dir / 'temporal_patterns.png')
        plt.close()

def main():
    """Main function to run the analysis"""
    # Load data
    df = load_burglary_data()
    
    # Perform EDA
    df = perform_eda(df)
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Calculate effectiveness metrics
    df = calculate_effectiveness_metrics(df)
    
    # Create effectiveness visualizations
    create_effectiveness_visualizations(df)
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()
