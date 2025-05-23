import os
import shutil
from pathlib import Path

def extract_boundaries():
    # Create output directory
    output_dir = Path('extracted_boundaries')
    output_dir.mkdir(exist_ok=True)
    
    # Create directories for City of London and Metropolitan Police
    city_london_dir = output_dir / 'city-of-london'
    met_police_dir = output_dir / 'metropolitan'
    city_london_dir.mkdir(exist_ok=True)
    met_police_dir.mkdir(exist_ok=True)
    
    # Process each year
    for year in ['2022', '2023', '2024', '2025']:
        year_dir = output_dir / year
        if not year_dir.exists():
            continue
            
        print(f"\nProcessing year: {year}")
        
        # Create year directories in output
        year_city_dir = city_london_dir / year
        year_met_dir = met_police_dir / year
        year_city_dir.mkdir(exist_ok=True)
        year_met_dir.mkdir(exist_ok=True)
        
        # Process each month
        for month_dir in year_dir.iterdir():
            if not month_dir.is_dir():
                continue
                
            month = month_dir.name
            print(f"Processing {month}...")
            
            # Create month directories
            month_city_dir = year_city_dir / month
            month_met_dir = year_met_dir / month
            month_city_dir.mkdir(exist_ok=True)
            month_met_dir.mkdir(exist_ok=True)
            
            # Copy City of London files
            city_source = month_dir / 'city-of-london'
            if city_source.exists():
                for file in city_source.glob('*'):
                    if file.is_file():
                        shutil.copy2(file, month_city_dir)
                print(f"  Copied City of London files from {month}")
            
            # Copy Metropolitan Police files
            met_source = month_dir / 'metropolitan'
            if met_source.exists():
                for file in met_source.glob('*'):
                    if file.is_file():
                        shutil.copy2(file, month_met_dir)
                print(f"  Copied Metropolitan Police files from {month}")

if __name__ == '__main__':
    print("Starting boundary extraction...")
    extract_boundaries()
    print("\nExtraction complete!") 