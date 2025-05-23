import geopandas as gpd
import pandas as pd
import numpy as np

def calculate_missing_areas():
    # Read the ESRI shapefile
    print("Reading ward boundaries from ESRI shapefile...")
    wards_gdf = gpd.read_file('statistical-gis-boundaries-london/London-wards-2018_ESRI/London_Ward.shp')
    
    # Convert to a more appropriate projection for area calculation (British National Grid)
    wards_gdf = wards_gdf.to_crs(epsg=27700)
    
    # Calculate areas in square kilometers
    wards_gdf['area_km2'] = wards_gdf.geometry.area / 1000000
    
    # Create a dictionary of ward areas
    ward_areas = dict(zip(wards_gdf['GSS_CODE'], wards_gdf['area_km2']))
    
    # Load existing ward areas
    print("\nLoading existing ward areas...")
    existing_areas = pd.read_csv('output_csv_files/ward_burglary_density.csv')
    
    # Find wards with missing areas
    missing_wards = existing_areas[existing_areas['Ward_Area'].isna()]
    print(f"\nFound {len(missing_wards)} wards with missing areas")
    
    # Try to find matches in the new data
    found_count = 0
    for idx, row in missing_wards.iterrows():
        ward_id = row['Ward ID']
        if ward_id in ward_areas:
            existing_areas.loc[idx, 'Ward_Area'] = ward_areas[ward_id]
            found_count += 1
    
    print(f"\nUpdated areas for {found_count} wards")
    
    # Recalculate crime density for updated wards
    existing_areas['Crime_Density'] = existing_areas['LSOA_Burglary_Count'] / existing_areas['Ward_Area']
    
    # Save updated data
    print("\nSaving updated ward areas...")
    existing_areas.to_csv('output_csv_files/ward_burglary_density_updated.csv', index=False)
    
    # Print summary of remaining missing areas
    still_missing = existing_areas[existing_areas['Ward_Area'].isna()]
    print(f"\nStill missing areas for {len(still_missing)} wards")
    if len(still_missing) > 0:
        print("\nWards still missing area data:")
        print(still_missing[['Ward ID', 'Ward Name', 'LSOA_Burglary_Count']].to_string())

if __name__ == "__main__":
    calculate_missing_areas() 