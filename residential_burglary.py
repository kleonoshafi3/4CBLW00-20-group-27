import pandas as pd

# 1. Read in your main burglary file and the unique‐locations list
df = pd.read_csv("output_csv_files/burglary_cases_with_ward_cleaned.csv")
unique = (
    pd.read_csv("output_csv_files/unique_locations.csv")
      .iloc[:, 0]  # first (and only) column: Unique_Location
      .str.replace(r"^On or near\s*", "", regex=True)
      .str.strip()
)

# 2. Build your non‐residential set by picking out every
#    “pure” place name (no street name) that is not a dwelling
#    or outbuilding per the ONS list:
non_residential = {
    "Parking Area",
    "Supermarket",
    "Police Station",
    "Bus/Coach Station",
    "Car Park",
    "Public Footpath",
    "Public House",
    "Park",
    "Office",
    "Shops",
    "Shop",
    "Factory",
    "Warehouse",
    "Restaurant",
    "Hotel",
    "Hospital",
    "Clinic",
    "Sports/Recreation Area",
    "Shopping Area",
    "Motorway Service Area",
    "Petrol Station",
    "Further/Higher Educational Building",
    "Theatre/Concert Hall",
    # …and any others you spot in `unique` that are clearly non-domestic
}

# 3. Strip the “On or near” prefix in your main DataFrame
df["Premises"] = (
    df["Location"]
      .str.replace(r"^On or near\s*", "", regex=True)
      .str.strip()
)

# 4. Filter: keep everything that isn’t in that non-residential set
df_residential = df[~df["Premises"].isin(non_residential)]
df_nonres     = df[df["Premises"].isin(non_residential)]

# 5. Save out for audit and for your pipeline
df_residential.to_csv("output_csv_files/residential_burglaries.csv", index=False)
df_nonres.to_csv("output_csv_files/filtered_out_non_residential.csv", index=False)
