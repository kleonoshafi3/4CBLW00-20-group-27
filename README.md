# Project Workflow

## Overview
This project involves extracting ward boundaries, calculating crime density, and analyzing missing ward data.

## Workflow

### 1. Extract Boundaries
- **Script**: `extract_boundaries.py`
- **Purpose**: Extracts ward boundaries from the `boundaries.zip` folder and saves them into the `extracted_boundaries` folder.
- **Process**: The script creates directories for each year (2022, 2023, 2024, 2025) and month, then copies files from the `city-of-london` and `metropolitan` subdirectories into the `extracted_boundaries` folder.

### 2. Calculate Crime Density
- **Script**: `data_loader_density.py`
- **Purpose**: Loads ward areas from the `extracted_boundaries` folder and calculates crime density.
- **Process**:
  - Extracts ward areas from KML files in the `extracted_boundaries` folder.
  - If any ward area is missing, it looks for the data in the `statistical-gis-boundaries-london` folder.
  - Generates two CSV files:
    - `output_csv_files/missing_wards.csv`: Lists wards with missing area data.
  - Calculates crime density for wards with complete area data and saves the results to `output_csv_files/ward_burglary_density.csv`.
  - Creates a temporal analysis of crime density and saves it to `output_csv_files/ward_temporal_analysis.csv`.
  - **Note**: The file `output_csv_files/burglary_cases_with_ward_cleaned.csv` is manually added to the project to facilitate data loading for `data_loader_density.py`.

### 3. Filter Missing Wards in London
- **Script**: `filter_missing_wards_in_london.py`
- **Purpose**: Filters the `missing_wards.csv` file to include only wards with missing area data that are located within London.
- **Process**: The script reads the `missing_wards.csv` file, checks against the London wards shapefile, and saves the filtered results to `output_csv_files/missing_wards_in_london.csv`.

## Summary
- **extract_boundaries.py** extracts boundary files from `boundaries.zip` and saves them into `extracted_boundaries`.
- **data_loader_density.py** uses these extracted files to calculate ward areas and crime density, generating CSV files for missing wards and crime density analysis.
- **filter_missing_wards_in_london.py** ensures that only wards with missing area data within London are included in the output file.

## Visualizations
- **extract_boundaries.py**: No visualizations generated.
- **data_loader_density.py**: Generates plots for crime density per ward and temporal trends in crime density (saved as PNG files in the `visualizations/` directory).
- **filter_missing_wards_in_london.py**: No visualizations generated.

## Requirements

- Python 3.x
- Required packages are listed in `requirements.txt`.

## Usage

1. Run `extract_boundaries.py` to generate the `extracted_boundaries` folder.
2. Run `data_loader_density.py` to load ward areas and generate missing ward reports.
3. Run `filter_missing_wards_in_london.py` to verify that all wards with missing areas belong outside of London.
