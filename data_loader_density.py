import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pykml import parser
from os import path
import os
from datetime import datetime
import geopandas as gpd
import re
from pathlib import Path

def calculate_ward_area(kml_file):
    """Calculate the area of a ward from its KML file."""
    print(f"Calculating area for {kml_file}")
    with open(kml_file) as f:
        doc = parser.parse(f)
        root = doc.getroot()
        # Get the coordinates from the KML file
        coords = root.Document.Placemark.Polygon.outerBoundaryIs.LinearRing.coordinates.text.strip().split()
        # Convert coordinates to list of (lon, lat) tuples
        coords = [tuple(map(float, coord.split(','))) for coord in coords]
        # Calculate area using the shoelace formula
        area = 0
        for i in range(len(coords)):
            j = (i + 1) % len(coords)
            area += coords[i][0] * coords[j][1]
            area -= coords[j][0] * coords[i][1]
        area = abs(area) / 2
        # Convert to square kilometers (approximate)
        area_km2 = area * 111.32 * 111.32 * np.cos(np.radians(coords[0][1]))
        print(f"Calculated area: {area_km2:.2f} km²")
        return area_km2

def extract_areas_from_kml_directories():
    """Extract ward areas from the extracted KML files"""
    print("\n=== Loading Areas from KML Files ===")
    
    ward_areas = {}
    total_files_processed = 0
    
    # Base directory for extracted boundaries
    base_dir = 'extracted_boundaries'
    
    # Process each year
    for year_dir in Path(base_dir).iterdir():
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
            
        year = year_dir.name
        print(f"\nProcessing year {year}...")
        
        # Process each month
        for month_dir in year_dir.iterdir():
            if not month_dir.is_dir():
                continue
                
            month = month_dir.name
            print(f"Processing {month}...")
            
            # Process each police force directory
            for force_dir in month_dir.iterdir():
                if not force_dir.is_dir():
                    continue
                    
                # Only process City of London and Metropolitan Police
                if force_dir.name not in ['city-of-london', 'metropolitan']:
                    continue
                    
                print(f"Processing {force_dir.name}...")
                
                # Process each KML file in the force directory
                for kml_file in force_dir.glob('*.kml'):
                    ward_id = kml_file.stem  # Get filename without extension
                    try:
                        area = calculate_ward_area(kml_file)
                        ward_areas[ward_id] = area
                        total_files_processed += 1
                    except Exception as e:
                        print(f"Error processing {kml_file}: {str(e)}")
    
    print(f"\nTotal: Extracted {len(ward_areas)} unique ward areas from {total_files_processed} KML files")
    return ward_areas

# Initialize ward_areas dictionary
ward_areas = {}

# Extract areas from KML files
print("\nStarting area extraction process...")
kml_areas = extract_areas_from_kml_directories()
print(f"Number of areas found from KML files: {len(kml_areas)}")
print("Sample of KML areas:", dict(list(kml_areas.items())[:5]))

# Use KML areas
ward_areas = kml_areas
print(f"\nTotal unique ward areas loaded: {len(ward_areas)}")
print("Sample of final areas:", dict(list(ward_areas.items())[:5]))

# 1. Load and Display Data
print("\nLoading crime data...")
try:
    df = pd.read_csv('output_csv_files/burglary_cases_with_ward_cleaned.csv')
    print("Successfully loaded crime data")
except Exception as e:
    print(f"Error loading crime data: {str(e)}")
    exit(1)

# Convert Month to datetime
df['Month'] = pd.to_datetime(df['Month'])

# Display first 5 rows
print("\nFirst 5 rows of the data:")
print(df.head())

# Display basic information about the dataset
print("\nDataset information:")
print(df.info())

# Create a mapping of Ward ID to Ward Name
ward_id_to_name = df[['Ward ID', 'Ward Name']].drop_duplicates().set_index('Ward ID')['Ward Name'].to_dict()

# 1. Count burglaries per LSOA within each ward
lsoa_counts = df.groupby(['Ward ID', 'LSOA code']).size().reset_index(name='LSOA_Burglary_Count')

# 2. Sum LSOA counts for each ward
ward_burglary_counts = lsoa_counts.groupby('Ward ID')['LSOA_Burglary_Count'].sum().reset_index()

# Add the Ward Name back for display purposes
ward_burglary_counts['Ward Name'] = ward_burglary_counts['Ward ID'].map(ward_id_to_name)

# Fallback: For missing wards, check statistical GIS boundaries
print("\nChecking statistical GIS boundaries for missing ward areas...")
try:
    # Read the ESRI shapefile
    wards_gdf = gpd.read_file('statistical-gis-boundaries-london/London-wards-2018_ESRI/London_Ward.shp')
    # Project to British National Grid for accurate area calculation
    wards_gdf = wards_gdf.to_crs(epsg=27700)
    wards_gdf['area_km2'] = wards_gdf.geometry.area / 1_000_000
    # Create a dictionary of ward areas from shapefile
    gis_ward_areas = dict(zip(wards_gdf['GSS_CODE'], wards_gdf['area_km2']))
    # Find missing ward IDs
    missing_ward_ids = set(ward_burglary_counts['Ward ID']) - set(ward_areas.keys())
    found_count = 0
    for ward_id in missing_ward_ids:
        if ward_id in gis_ward_areas:
            ward_areas[ward_id] = gis_ward_areas[ward_id]
            found_count += 1
    print(f"Added area data for {found_count} missing wards from GIS boundaries.")
    print(f"Total unique ward areas after fallback: {len(ward_areas)}")
except Exception as e:
    print(f"Error loading GIS boundaries for fallback: {e}")

# After processing all directories, check how many wards still have missing areas
remaining_missing = ward_burglary_counts[~ward_burglary_counts['Ward ID'].isin(ward_areas.keys())]
print(f"\n=== Area Coverage Analysis ===")
print(f"Total number of wards in crime data: {len(ward_burglary_counts)}")
print(f"Number of wards with area data: {len(ward_areas)}")
print(f"Number of wards with missing area data: {len(remaining_missing)}")
print(f"Coverage percentage: {(len(ward_areas) / len(ward_burglary_counts)) * 100:.2f}%")

# Save missing wards to a CSV file
missing_wards_path = 'output_csv_files/missing_wards.csv'
remaining_missing[['Ward ID', 'Ward Name', 'LSOA_Burglary_Count']].to_csv(missing_wards_path, index=False)
print(f"\nMissing wards saved to {missing_wards_path}")

if len(remaining_missing) > 0:
    print("\nSample of wards with missing area data:")
    print(remaining_missing[['Ward ID', 'Ward Name', 'LSOA_Burglary_Count']].head().to_string())
    
    # Print unique ward IDs that are missing
    print("\nUnique ward IDs missing area data:")
    print(sorted(remaining_missing['Ward ID'].unique()))

# Create a flag to indicate whether an area value exists
ward_burglary_counts['Has_Area_Data'] = ward_burglary_counts['Ward ID'].isin(ward_areas.keys())

# 4. Add area and calculate density only for wards with area data
ward_burglary_counts['Ward_Area'] = ward_burglary_counts['Ward ID'].map(ward_areas)
ward_burglary_counts['Crime_Density'] = None  # Initialize as None
ward_burglary_counts.loc[ward_burglary_counts['Has_Area_Data'], 'Crime_Density'] = (
    ward_burglary_counts.loc[ward_burglary_counts['Has_Area_Data'], 'LSOA_Burglary_Count'] / 
    ward_burglary_counts.loc[ward_burglary_counts['Has_Area_Data'], 'Ward_Area']
)

# Convert Crime_Density to numeric type, replacing any non-numeric values with NaN
ward_burglary_counts['Crime_Density'] = pd.to_numeric(ward_burglary_counts['Crime_Density'], errors='coerce')

# Print summary statistics
print("\nSummary of results:")
print("\nWards with missing area data:")
missing_areas = ward_burglary_counts[~ward_burglary_counts['Has_Area_Data']]
print(f"Number of wards with missing area: {len(missing_areas)}")
if len(missing_areas) > 0:
    print("\nWards with missing area data:")
    print(missing_areas[['Ward ID', 'Ward Name', 'LSOA_Burglary_Count']].to_string())

print("\nWards with complete data:")
complete_data = ward_burglary_counts[ward_burglary_counts['Has_Area_Data']]
print(f"Number of wards with complete data: {len(complete_data)}")
print("\nSample of wards with complete data:")
print(complete_data[['Ward ID', 'Ward Name', 'LSOA_Burglary_Count', 'Ward_Area', 'Crime_Density']].head().to_string())

# 5. Create temporal analysis only for wards with area data
print("\nCreating temporal analysis...")
# Group by ward ID and month 
temporal_data = df.groupby(['Ward ID', 'Month']).size().reset_index(name='Monthly_Burglaries')
# Add Ward Name for readability
temporal_data['Ward Name'] = temporal_data['Ward ID'].map(ward_id_to_name)

# Filter for wards with area data
temporal_data = temporal_data[temporal_data['Ward ID'].isin(ward_areas.keys())]

# Add area for each ward to temporal_data
temporal_data['Ward_Area'] = temporal_data['Ward ID'].map(ward_areas)
# Calculate monthly crime density (if area is available)
temporal_data['Monthly_Crime_Density'] = temporal_data['Monthly_Burglaries'] / temporal_data['Ward_Area']
# Save the new temporal analysis with density (overwrite the previous file)
temporal_data.to_csv('output_csv_files/ward_temporal_analysis.csv', index=False)
print("\nSaved temporal crime density analysis to 'output_csv_files/ward_temporal_analysis.csv'")

# Drop rows with missing area values and overwrite the CSV file
ward_burglary_counts_no_missing = ward_burglary_counts.dropna(subset=['Ward_Area'])
ward_burglary_counts_no_missing.to_csv('output_csv_files/ward_burglary_density.csv', index=False)
print("\nOverwrote 'ward_burglary_density.csv' with only rows that have area values.")

print("\nAnalysis complete!")

# Create output directory for visualizations if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

# 5.1 Create line chart for temporal pattern
plt.figure(figsize=(15, 8))
# Plot average crime density over time
avg_density = temporal_data.groupby('Month')['Monthly_Burglaries'].mean()
plt.plot(avg_density.index, avg_density.values, marker='o')
plt.title('Average Crime Count Over Time')
plt.xlabel('Month')
plt.ylabel('Average Crime Count')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/crime_count_trend.png', dpi=300, bbox_inches='tight')
plt.close()

# 5.2 Create bar chart for top 5 wards
plt.figure(figsize=(15, 8))
top_wards = complete_data.nlargest(5, 'Crime_Density')

# Create labels that include both Ward ID and Ward Name
ward_labels = [f"{row['Ward ID']}\n{row['Ward Name']}" for _, row in top_wards.iterrows()]

# Use Ward ID + Ward Name as x-axis labels
plt.bar(range(len(top_wards)), top_wards['Crime_Density'])
plt.xticks(range(len(top_wards)), ward_labels, rotation=45, ha='right')

plt.title('Crime Density for Top 5 Wards', fontsize=16)
plt.xlabel('Ward (ID and Name)', fontsize=14)
plt.ylabel('Crime Density (crimes per sq km)', fontsize=14)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Add value labels on top of each bar
for i, v in enumerate(top_wards['Crime_Density']):
    plt.text(i, v + 10, f"{v:.1f}", ha='center', fontsize=12)

# Add a subtitle with additional context
plt.figtext(0.5, 0.01, "Ward ID shown above Ward Name to ensure unique identification", 
            ha="center", fontsize=10, style='italic')

plt.savefig('visualizations/top_5_wards.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nVisualizations saved to 'visualizations' directory:")
print("- crime_count_trend.png: Line chart showing how average crime count changes over time")
print("- top_5_wards.png: Bar chart showing the 5 wards with highest crime density") 