import pandas as pd
import geopandas as gpd

# Load missing wards
missing_wards = pd.read_csv('output_csv_files/missing_wards.csv')

# Load London wards shapefile
london_gdf = gpd.read_file('statistical-gis-boundaries-london/London-wards-2018_ESRI/London_Ward.shp')

# Get all valid London ward IDs from the shapefile
london_ward_ids = set(london_gdf['GSS_CODE'])

# Filter missing wards to only those inside London
missing_wards_in_london = missing_wards[missing_wards['Ward ID'].isin(london_ward_ids)]

# Save the filtered result
missing_wards_in_london.to_csv('output_csv_files/missing_wards_in_london.csv', index=False)
print(f"Filtered missing wards saved to output_csv_files/missing_wards_in_london.csv") 