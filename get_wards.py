import json
import pandas as pd
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
uk_fp    = DATA_DIR / "UK_wards_2022.geojson"
csv_fp   = DATA_DIR / "burglary_cases_with_ward_cleaned.csv"
out_fp   = DATA_DIR / "wards_london_crime_2024.geojson"
# ── 1) Load all ward-codes from your burglaries CSV ─────────────────
wards_df = pd.read_csv(csv_fp, usecols=["Ward ID"])
ward_ids = set(wards_df["Ward ID"].dropna().unique())

# ── 2) Define the 32 boroughs + City exactly as per LAD22NM ─────────
london_LAs = {
    "City of London",
    "Barking and Dagenham","Barnet","Bexley","Brent","Bromley",
    "Camden","Croydon","Ealing","Enfield","Greenwich","Hackney",
    "Hammersmith and Fulham","Haringey","Harrow","Havering",
    "Hillingdon","Hounslow","Islington","Kensington and Chelsea",
    "Kingston upon Thames","Lambeth","Lewisham","Merton","Newham",
    "Redbridge","Richmond upon Thames","Southwark","Sutton",
    "Tower Hamlets","Waltham Forest","Wandsworth", "Westminster"
}

# ── 3) Load the UK wards GeoJSON (ensure UTF-8!) ────────────────────
with open(uk_fp, "r", encoding="utf-8") as f:
    uk = json.load(f)

# ── 4) Filter to only those features that satisfy both conditions ────
london_feats = [
    feat for feat in uk["features"]
    if (
        feat["properties"].get("WD22CD") in ward_ids
        and feat["properties"].get("LAD22NM") in london_LAs
    )
]

# ── 5) Write out the trimmed London wards file ──────────────────────
london_geo = {
    "type": "FeatureCollection",
    "features": london_feats
}
with open(out_fp, "w", encoding="utf-8") as f:
    json.dump(london_geo, f)

print(f"✅ Created {out_fp.name} with {len(london_feats)} wards")
