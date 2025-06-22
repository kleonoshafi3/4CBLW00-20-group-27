import pandas as pd
from pathlib import Path
import json
import dash
from dash import dcc, html, Input, Output
from dash import dash_table
from dash import callback_context
import dash_bootstrap_components as dbc
import plotly.express as px

OUTPUT_DIR = Path(__file__).parent / "output_csv_files"
# ---- 1. LOAD & PREPARE ----
def load_and_prepare(output_dir: str):
    data_path = Path(output_dir)

    # 1) Read the single cleaned CSV
    fp = data_path / "burglary_cases_with_ward_cleaned.csv"
    full = pd.read_csv(
        fp,
        parse_dates=["Month"],  # parse Month into datetime
        dtype={"Ward ID": str, "Ward Name": str}
    )

    # 2) Rename it to MonthDt for consistency
    full["MonthDt"] = full["Month"]

    # 3) Filter down to burglaries only
    full = full[
        full["Crime type"].str.strip().str.lower() == "burglary"
        ].copy()

    return full
# ---- MAIN DATAFRAME ----
DATA_DIR = Path(__file__).parent / "data"
df = load_and_prepare(str(DATA_DIR))
last_month = df["MonthDt"].max()
last_updated_str = last_month.strftime("%B %Y")
assert "Ward Name" in df.columns, "No Ward Name column—did the merge fail?"
# derive borough name from LSOA name
df["Borough"] = df["LSOA name"].str.replace(r"\s+\d+[A-Z]?$", "", regex=True)
assert df["Crime type"].str.strip().str.lower().eq("burglary").all(), "Non-burglary records found!"
if df.empty:
    raise ValueError("❗ No burglary records found.")
# load boroughs GeoJSON
with open(DATA_DIR / "boroughs_london.geojson") as f:
    boroughs_geo = json.load(f)

#load wards GeoJSON
with open(DATA_DIR / "wards_london_crime_2024.geojson", encoding="utf-8") as f:
    wards_geo = json.load(f)

# slider bounds
years = sorted(df["MonthDt"].dt.year.dropna().astype(int).unique().tolist())
months = sorted(df["MonthDt"].dt.month.dropna().astype(int).unique().tolist())
year_marks = {y: str(y) for y in years}
month_marks = {m: str(m) for m in months}

imd_fp = DATA_DIR / "imd_change_2007_2019_spatial.csv"
imd_df = pd.read_csv(
    imd_fp,
    dtype={"ward_code": str, "ward_name": str, "imd_est_2024": float}
).rename(columns={
    "ward_code": "Ward ID",
    "ward_name": "Ward Name"
})
# ---- LOAD PREDICTED OFFICERS ----
# 1) read the raw predictions
pred_fp = Path(DATA_DIR) / "predicted_officers.csv"
pred_df = pd.read_csv(
    pred_fp,
    dtype={
      "Ward ID": str,
      "Time Slot": str,
      "Day Type": str,
      "Month": int,
      "Year": int,
      "Officers": int
    }
)

# 2) read the ward‐code → ward_name lookup
ward_map_fp = OUTPUT_DIR / "london_wards_2024.csv"
ward_map = (
    pd.read_csv(ward_map_fp, dtype={"ward_code": str, "ward_name": str})
      .rename(columns={"ward_code": "Ward ID", "ward_name": "Ward Name"})
)

# 3) merge so every row in pred_df has a human‐readable ward name
pred_df = pred_df.merge(
    ward_map[["Ward ID", "Ward Name"]],
    on="Ward ID",
    how="left"
)
# 4) bring in Borough from the burglary df to filter by it
ward_to_boro = (
    df[["Ward ID","Borough"]]
    .drop_duplicates(subset=["Ward ID","Borough"])
)
pred_df = pred_df.merge(ward_to_boro, on="Ward ID", how="left")



# ---- DASH APP LAYOUT ----
app = dash.Dash(
    __name__,
    title="London Burglary Dashboard",
    external_stylesheets=[dbc.themes.FLATLY]
)

valid_wards = {feat["properties"]["WD24CD"] for feat in wards_geo["features"]}
# ---- FILTER CONTROLS  ----
filter_controls = [

    # Location
    html.Div([
        html.Label("Select Location(s)", htmlFor="location-dropdown", className="form-label"),
        dcc.Dropdown(
            id="location-dropdown",
            options=[{"label": loc, "value": loc} for loc in sorted(df["Location"].dropna().unique())],
            value=[],
            multi=True,
            placeholder="Select locations..."
        ),
        dbc.Spinner(
            dash_table.DataTable(
                id="location-table",
                columns=[{"name":"Location","id":"Location"},{"name":"Count","id":"Count"}],
                data=[],
                style_table={"height":"150px","overflowY":"auto"}
            ),
            color="secondary", size="sm"
        )
    ], className="mb-3"),

    # LSOA
    html.Div([
        html.Label("Select LSOA area(s)", htmlFor="lsoa-dropdown", className="form-label"),
        dcc.Dropdown(
            id="lsoa-dropdown",
            options=[{"label": l, "value": l} for l in sorted(df["LSOA name"].dropna().unique())],
            value=[],
            multi=True,
            placeholder="Select LSOAs..."
        ),
        dbc.Spinner(
            dash_table.DataTable(
                id="lsoa-table",
                columns=[{"name":"LSOA name","id":"LSOA name"},{"name":"Count","id":"Count"}],
                data=[],
                style_table={"height":"150px","overflowY":"auto"}
            ),
            color="secondary", size="sm"
        )
    ], className="mb-3"),

    # Borough
    html.Div([
        html.Label("Select Borough(s)", htmlFor="borough-dropdown", className="form-label"),
        dcc.Dropdown(
            id="borough-dropdown",
            options=[{"label": b, "value": b} for b in sorted(df["Borough"].dropna().unique())],
            value=[],
            multi=True,
            placeholder="Select boroughs..."
        ),
        dbc.Spinner(
            dash_table.DataTable(
                id="borough-table",
                columns=[{"name":"Borough","id":"Borough"},{"name":"Count","id":"Count"}],
                data=[],
                style_table={"height":"150px","overflowY":"auto"}
            ),
            color="secondary", size="sm"
        )
    ], className="mb-3"),

    # Ward
    html.Div([
        html.Label("Select Ward(s)", htmlFor="ward-dropdown", className="form-label"),
        dcc.Dropdown(
            id="ward-dropdown",
            options=[{"label": w, "value": w} for w in sorted(df["Ward Name"].dropna().unique())],
            value=[],
            multi=True,
            placeholder="Select wards..."
        ),
        dbc.Spinner(
            dash_table.DataTable(
                id="ward-table",
                columns=[{"name":"Ward","id":"Ward Name"},{"name":"Count","id":"Count"}],
                data=[],
                style_table={"height":"150px","overflowY":"auto"}
            ),
            color="secondary", size="sm"
        )
    ], className="mb-3"),

    # Outcome
    html.Div([
        html.Label("Select Outcome Category(ies)", htmlFor="outcome-dropdown", className="form-label"),
        dcc.Dropdown(
            id="outcome-dropdown",
            options=[{"label": o, "value": o} for o in sorted(df["Last outcome category"].dropna().unique())],
            value=[],
            multi=True,
            placeholder="Filter by outcome..."
        ),
        dbc.Spinner(
            dash_table.DataTable(
                id="outcome-table",
                columns=[{"name":"Outcome Category","id":"Last outcome category"},{"name":"Count","id":"Count"}],
                data=[],
                style_table={"height":"150px","overflowY":"auto"}
            ),
            color="secondary", size="sm"
        )
    ], className="mb-3"),

    # Date Range
    html.Div([
        html.Label("Select Date Range", htmlFor="date-picker", className="form-label"),
        dcc.DatePickerRange(
            id="date-picker",
            min_date_allowed=pd.to_datetime("2022-03-01"),
            max_date_allowed=pd.to_datetime("2025-02-28"),
            start_date=pd.to_datetime("2022-03-01"),
            end_date=pd.to_datetime("2025-02-01"),
            display_format="YYYY-MM",
            month_format="YYYY-MM",
            with_portal=True
        ),
        dbc.Tooltip("Choose your start and end month (Mar 2022 → Feb 2025).",
                    target="date-picker", placement="right"),
        html.Div(id="date-error", style={"color":"red","fontWeight":"bold"})
    ], className="mb-3"),

    # Year Checklist
    html.Div([
        html.Label("Select Year(s)", htmlFor="year-checklist", className="form-label"),
        dcc.Checklist(
            id="year-checklist",
            options=[{"label": str(y), "value": y} for y in years],
            value=[],
            inline=True,
            labelStyle={"display":"inline-block","margin-right":"1rem"}
        )
    ], className="mb-3"),

    # Month Checklist
    html.Div([
        html.Label("Select Month(s)", htmlFor="month-of-year-checklist", className="form-label"),
        dcc.Checklist(
            id="month-of-year-checklist",
            options=[
              {"label":"Jan","value":1}, {"label":"Feb","value":2}, {"label":"Mar","value":3},
              {"label":"Apr","value":4}, {"label":"May","value":5}, {"label":"Jun","value":6},
              {"label":"Jul","value":7}, {"label":"Aug","value":8}, {"label":"Sep","value":9},
              {"label":"Oct","value":10},{"label":"Nov","value":11},{"label":"Dec","value":12},
            ],
            value=[],
            inline=True,
            labelStyle={"display":"inline-block","margin-right":"1rem"}
        )
    ], className="mb-3"),

    # Force Checklist
    html.Div([
        html.Label("Select Force(s)", htmlFor="force-checklist", className="form-label"),
        dcc.Checklist(
            id="force-checklist",
            options=[{"label": f, "value": f} for f in df["Reported by"].unique()],
            value=df["Reported by"].unique().tolist(),
            inline=True,
            labelStyle={"display":"inline-block","margin-right":"1rem"}
        ),
        dbc.Tooltip("Filter by which police force reported the crime.",
                    target="force-checklist", placement="right")
    ], className="mb-3"),

    # View RadioItems
    html.Div([
        html.Label("Select View", htmlFor="view-selector", className="form-label"),
        dcc.RadioItems(
            id="view-selector",
            options=[
                {"label":"Points","value":"points"},
                {"label":"LSOA bubbles","value":"lsoa_bubbles"},
                {"label":"Borough heat","value":"boroughs"},
                {"label":"Ward heat","value":"wards"},
                {"label":"IMD est. 2024", "value":"imd"},
                {"label":"Officer allocation","value":"officers"},
                {"label":"Crime density","value":"density"},
            ],
            value="wards",
            inline=True,
            labelStyle={"display":"inline-block","margin-right":"1rem"}
        ),
        dbc.Tooltip("Choose how crimes are visualized on the map.",
                    target="view-selector", placement="right")
    ], className="mb-3"),

    # Reset Button
    html.Div([
        dbc.Button("🔄 Reset Filters", id="reset-button",
                   color="secondary", className="w-100 mt-2 mb-3", n_clicks=0)
    ], className="mb-3"),
    # Day Type
    html.Div([
        html.Label("Select Day Type", htmlFor="daytype-dropdown", className="form-label"),
        dcc.RadioItems(
            id="daytype-dropdown",
            options=[
                {"label": "Weekday", "value": "Weekday"},
                {"label": "Weekend", "value": "Weekend"}
            ],
            value=None,
            inline=True,
            labelStyle={"display": "inline-block", "margin-right": "1rem"}
        )
    ], className="mb-3"),

    # Time Slot
    html.Div([
        html.Label("Select Time Slot", htmlFor="timeslot-dropdown", className="form-label"),
        dcc.Dropdown(
            id="timeslot-dropdown",
            options=[{"label": ts, "value": ts} for ts in sorted(pred_df["Time Slot"].unique())],
            value=None,
            multi=False,
            placeholder="All time slots"
        )
    ], className="mb-3"),
]
app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <!-- ← this loads 'Inter' from Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""
server = app.server
app.layout = dbc.Container(fluid=True, children=[

  # ——— PAGE TITLE ——————————————————————————————————————————————————————
  html.H1("London Burglary Dashboard", className="text-center my-4"),

  # ——— STICKY KEY METRICS BAR —————————————————————————————————————————————
  html.Div(
      dbc.Row([
          dbc.Col(dbc.Card([dbc.CardHeader("Total Burglaries"),
                              dbc.CardBody(html.H3(id="total-count"))], body=True), width=4),
          dbc.Col(dbc.Card([dbc.CardHeader("Distinct Wards"),
                              dbc.CardBody(html.H3(id="ward-count"))], body=True), width=4),
          dbc.Col(dbc.Card([dbc.CardHeader("Avg / Month"),
                              dbc.CardBody(html.H3(id="avg-month"))], body=True), width=4),
      ], className="mb-4"),
      style={
          "position": "sticky",
          "top": 0,
          "zIndex": 2000,
          "backgroundColor": "white",
          "paddingTop": "1rem",
          "paddingBottom": "0.5rem",
          "borderBottom": "1px solid #ddd"
      }
  ),

    # ——— FILTER SIDEBAR + MAP —————————————————————————————————————————————
  dbc.Row([

    # • Sidebar (collapsible)
    dbc.Col(
      html.Div([
        dbc.Button("Filters 🔧", id="toggle-btn", color="primary", className="mb-3", n_clicks=0),
        dbc.Collapse(
          dbc.Card(
            dbc.CardBody([
              html.H5("Filter Burglaries", className="mb-3"),
              *filter_controls
            ]),
            className="shadow-sm"
          ),
          id="sidebar-collapse",
          is_open=True
        )
      ],
      style={
        "height": "90vh",        
        "overflowY": "auto",     
        "paddingRight": "1rem"   
      }
      ),
      width=3
    ),

    # • Map
    dbc.Col(
      dbc.Spinner(
        dcc.Graph(id="map-graph", style={"height":"80vh"}),
        color="primary",
        delay_show=4000
      ),
      width=9
    ),

  ]),  # end filter+map row

  # ——— FOOTER —————————————————————————————————————————————————————
  html.Footer([
    html.Div([
      "Data source: ",
      html.A("London DataStore", href="https://data.london.gov.uk/", target="_blank")
    ], className="text-center mb-1"),
    html.Div(f"Last updated: {last_updated_str}", className="text-center")
  ])

])
@app.callback(
    Output("sidebar-collapse", "is_open"),
    Input("toggle-btn", "n_clicks"),
    prevent_initial_call=True
)
def toggle_sidebar(n):
    return n % 2 == 1

# ---- CALLBACK: populate tables ----
@app.callback(
    [
      Output("location-table", "data"),
      Output("lsoa-table",      "data"),
      Output("borough-table",   "data"),
      Output("ward-table",      "data"),
      Output("outcome-table",   "data")
    ],
    [
      Input("location-dropdown", "value"),
      Input("lsoa-dropdown",     "value"),
      Input("borough-dropdown",  "value"),
      Input("ward-dropdown",     "value"),
      Input("date-picker",  "start_date"),
      Input("date-picker",    "end_date"),
      Input("year-checklist",    "value"),
      Input("month-of-year-checklist", "value"),
      Input("force-checklist",   "value"),
      Input("outcome-dropdown",  "value")
    ]
)
def update_tables(loc_vals, lsoa_vals, borough_vals, ward_vals, start_date, end_date, yols, moys, forces, outcomes):
    d = df.copy()
    if loc_vals: d = d[d["Location"].isin(loc_vals)]
    if lsoa_vals: d = d[d["LSOA name"].isin(lsoa_vals)]
    if borough_vals: d = d[d["Borough"].isin(borough_vals)]
    if ward_vals:   d = d[d["Ward Name"].isin(ward_vals)]
    if outcomes: d = d[d["Last outcome category"].isin(outcomes)]
    if yols:
        d = d[d["MonthDt"].dt.year.isin(yols)]
    if moys:
        d = d[d["MonthDt"].dt.month.isin(moys)]
    # date filter
    d = d[(d["MonthDt"] >= pd.to_datetime(start_date)) &
          (d["MonthDt"] <= pd.to_datetime(end_date)) &
          d["Reported by"].isin(forces)]
    loc_df = d.groupby("Location").size().reset_index(name="Count").sort_values(by="Count", ascending=False)
    lsoa_df = d.groupby("LSOA name").size().reset_index(name="Count").sort_values(by="Count", ascending=False)
    borough_df = d.groupby("Borough").size().reset_index(name="Count").sort_values(by="Count", ascending=False)
    ward_df = d.groupby("Ward Name").size().reset_index(name="Count").sort_values("Count", ascending=False)
    outcome_df = d.groupby("Last outcome category").size().reset_index(name="Count").sort_values(by="Count", ascending=False)
    return loc_df.to_dict('records'), lsoa_df.to_dict('records'), borough_df.to_dict('records'), ward_df.to_dict('records'), outcome_df.to_dict('records')

from dash import Input, Output 

@app.callback(
    [
        Output("total-count", "children"),
        Output("ward-count", "children"),
        Output("avg-month",    "children")
    ],
    [
        Input("location-dropdown",        "value"),
        Input("lsoa-dropdown",            "value"),
        Input("borough-dropdown",         "value"),
        Input("ward-dropdown",            "value"),
        Input("date-picker",              "start_date"),
        Input("date-picker",              "end_date"),
        Input("year-checklist",           "value"),
        Input("month-of-year-checklist",  "value"),
        Input("force-checklist",          "value"),
        Input("outcome-dropdown",         "value")
    ]
)
def update_metrics(loc_vals, lsoa_vals, borough_vals, ward_vals,
                   start_date, end_date, yols, moys, forces, outcomes):
    d = df.copy()
    if loc_vals:     d = d[d["Location"].isin(loc_vals)]
    if lsoa_vals:    d = d[d["LSOA name"].isin(lsoa_vals)]
    if borough_vals: d = d[d["Borough"].isin(borough_vals)]
    if ward_vals:    d = d[d["Ward Name"].isin(ward_vals)]
    if outcomes:     d = d[d["Last outcome category"].isin(outcomes)]
    if yols:         d = d[d["MonthDt"].dt.year.isin(yols)]
    if moys:         d = d[d["MonthDt"].dt.month.isin(moys)]
    # date bounds + force filter
    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date)
    d = d[
        (d["MonthDt"] >= start) &
        (d["MonthDt"] <= end) &
        (d["Reported by"].isin(forces))
    ]
    total = len(d)
    ward_count = len(valid_wards & set(d["Ward ID"].dropna()))
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    months_span = (end.year - start.year) * 12 + (end.month - start.month) + 1
    avg_per_month = total / months_span if months_span > 0 else 0

    return f"{total:,}", f"{ward_count}", f"{avg_per_month:.1f}"

# ---- CALLBACK: dynamic dropdowns ----
@app.callback(
    Output("location-dropdown", "options"),
    [Input("borough-dropdown", "value"), Input("lsoa-dropdown", "value"), Input("ward-dropdown", "value"), Input("outcome-dropdown", "value")]
)
def update_location_options(boros, lsoas, wards, outcomes):
    d = df.copy()
    if boros: d = d[d["Borough"].isin(boros)]
    if lsoas: d = d[d["LSOA name"].isin(lsoas)]
    if wards:
        d = d[d["Ward Name"].isin (wards)]
    if outcomes:
        d = d[d["Last outcome category"].isin(outcomes)]
    return [{"label": loc, "value": loc} for loc in sorted(d["Location"].dropna().unique())]

@app.callback(
    Output("lsoa-dropdown", "options"),
    [Input("borough-dropdown", "value"), Input("location-dropdown", "value"), Input("ward-dropdown", "value"), Input("outcome-dropdown", "value")]
)
def update_lsoa_options(boros, locs, wards, outcomes):
    d = df.copy()
    if boros: d = d[d["Borough"].isin(boros)]
    if locs: d = d[d["Location"].isin(locs)]
    if wards:
        d = d[d["Ward Name"].isin (wards)]
    if outcomes:
        d = d[d["Last outcome category"].isin(outcomes)]
    return [{"label": lsoa, "value": lsoa} for lsoa in sorted(d["LSOA name"].dropna().unique())]

# ---- CALLBACK: dynamic borough options ----
@app.callback(
    Output("borough-dropdown", "options"),
    [Input("location-dropdown", "value"), Input("lsoa-dropdown", "value"), Input("ward-dropdown", "value"), Input("outcome-dropdown", "value")]
)
def update_borough_options(locs, lsoas, wards, outcomes):
    d = df.copy()
    if locs:
        d = d[d["Location"].isin(locs)]
    if lsoas:
        d = d[d["LSOA name"].isin(lsoas)]
    if wards:
        d = d[d["Ward Name"].isin (wards)]
    if outcomes:
        d = d[d["Last outcome category"].isin(outcomes)]
    return [{"label": b, "value": b} for b in sorted(d["Borough"].dropna().unique())]

# ---- CALLBACK: dynamic ward options
@app.callback(
    Output("ward-dropdown", "options"),
    [Input("location-dropdown", "value"), Input("lsoa-dropdown", "value"), Input("borough-dropdown", "value"), Input("outcome-dropdown", "value")]
)
def update_ward_options(locs, lsoas, boros, outcomes):
    d = df.copy()
    if locs:
        d = d[d["Location"].isin(locs)]
    if lsoas:
        d = d[d["LSOA name"].isin(lsoas)]
    if boros:
        d = d[d["Borough"].isin (boros)]
    if outcomes:
        d = d[d["Last outcome category"].isin(outcomes)]
    return [{"label": b, "value": b} for b in sorted(d["Ward Name"].dropna().unique())]

def update_ward_count(loc_vals, lsoa_vals, borough_vals, ward_vals,
                      start_date, end_date, yols, moys, forces, outcomes):
    # apply exactly the same filtering logic you use elsewhere…
    d = df.copy()
    if loc_vals:     d = d[d["Location"].isin(loc_vals)]
    if lsoa_vals:    d = d[d["LSOA name"].isin(lsoa_vals)]
    if borough_vals: d = d[d["Borough"].isin(borough_vals)]
    if ward_vals:    d = d[d["Ward Name"].isin(ward_vals)]
    if outcomes:     d = d[d["Last outcome category"].isin(outcomes)]
    if yols:         d = d[d["MonthDt"].dt.year.isin(yols)]
    if moys:         d = d[d["MonthDt"].dt.month.isin(moys)]

    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date)
    d = d[
      (d["MonthDt"] >= start) &
      (d["MonthDt"] <= end) &
      (d["Reported by"].isin(forces))
    ]
    valid = { feat["properties"]["WD24CD"]
              for feat in wards_geo["features"] }
    ward_count = len( valid & set(d["Ward ID"]) )
    return f"{ward_count}"

# ---- CALLBACK: dynamic outcome options ----
@app.callback(
    Output("outcome-dropdown", "options"),
    [Input("location-dropdown", "value"), Input("lsoa-dropdown", "value"), Input("borough-dropdown", "value"), Input("ward-dropdown", "value")]
)
def update_outcome_options(locs, lsoas, boros, wards):
    d = df.copy()
    if locs:
        d = d[d["Location"].isin(locs)]
    if lsoas:
        d = d[d["LSOA name"].isin(lsoas)]
    if boros:
        d = d[d["Borough"].isin(boros)]
    if wards:
        d = d[d["Ward Name"].isin(wards)]
    return [{"label": o, "value": o} for o in sorted(d["Last outcome category"].dropna().unique())]

# ---- CALLBACK: update map & count ----
@app.callback(
    [Output("map-graph", "figure"), Output("date-error", "children")],
    [
        Input("location-dropdown", "value"), Input("lsoa-dropdown", "value"), Input("borough-dropdown", "value"),
        Input("ward-dropdown","value"), Input("date-picker", "start_date"), Input("date-picker", "end_date"),
        Input("year-checklist","value"), Input("month-of-year-checklist", "value"), Input("force-checklist", "value"),
        Input("outcome-dropdown", "value"), Input("view-selector", "value"), Input("daytype-dropdown",  "value"),
        Input("timeslot-dropdown", "value"),Input("map-graph", "relayoutData"), Input("reset-button", "n_clicks"),
    ]
)
def update_dashboard(loc_vals, lsoa_vals, borough_vals, ward_vals, start_date, end_date, yols, moys, forces, outcomes, view, daytype, timeslot, relayout, reset_n):
    d = df.copy()
    if loc_vals: d = d[d["Location"].isin(loc_vals)]
    if lsoa_vals: d = d[d['LSOA name'].isin(lsoa_vals)]
    if borough_vals: d = d[d['Borough'].isin(borough_vals)]
    if ward_vals:    d = d[d["Ward Name"].isin(ward_vals)]
    if outcomes: d = d[d['Last outcome category'].isin(outcomes)]
    if yols:
        d = d[d["MonthDt"].dt.year.isin(yols)]
    if moys:
        d = d[d["MonthDt"].dt.month.isin(moys)]
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    if start_dt < pd.Timestamp(2022, 3, 1) or end_dt > pd.Timestamp(2025, 2, 28):
        return "", "", {}, "❗ Date must lie between Mar 2022 and Feb 2025"
    if start_dt > end_dt:
        return "", "", {}, "❗ Start must be before End"
    d = d[
        (d["MonthDt"] >= start_dt) &
        (d["MonthDt"] <= end_dt) &
        (d["Reported by"].isin(forces))
        ]
    count_text = f"Total burglaries: {len(d):,}"
    if view == "points":
        view_text = "View: Points"
        fig = px.scatter_mapbox(d, lat="Latitude", lon="Longitude",
                                hover_name="Crime ID",
                                hover_data=["Month", "Reported by", "Last outcome category", "Location"],
                                zoom=12, opacity=0.6, height=800,
                                mapbox_style="open-street-map")
    elif view == "lsoa_bubbles":
        view_text = "View: LSOA bubbles"
        # count & centroid per LSOA
        agg = (
            d.groupby(["LSOA code", "LSOA name"], as_index=False)
            .agg({
                "Crime ID": "size",
                "Latitude": "mean",
                "Longitude": "mean"
            })
            .rename(columns={"Crime ID": "Count"})
        )
        lsoa_to_ward = (
            d[["LSOA code", "Ward Name"]]
            .drop_duplicates(subset=["LSOA code", "Ward Name"])
            .groupby("LSOA code", as_index=False).first()
        )
        agg = agg.merge(lsoa_to_ward, on="LSOA code", how="left")
        fig = px.scatter_mapbox(
            agg,
            lat="Latitude",
            lon="Longitude",
            size="Count",
            color="Count",
            hover_name="LSOA name",
            hover_data=["Count", "Ward Name"], 
            size_max=30,
            height=800,
            mapbox_style="open-street-map",
            color_continuous_scale=px.colors.diverging.RdBu[::-1]
        )
    elif view == "imd":
        imd_plot = imd_df.copy()
        if ward_vals:
            imd_plot = imd_plot[imd_plot["Ward Name"].isin(ward_vals)]
        fig = px.choropleth_mapbox(
            imd_plot,
            geojson=wards_geo,
            locations="Ward ID",  
            featureidkey="properties.WD24CD", 
            color="imd_est_2024",
            hover_name="Ward Name",
            hover_data=["imd_est_2024"],
            mapbox_style="carto-positron",
            height=800,
            color_continuous_scale=px.colors.sequential.Plasma
        )
    elif view == "boroughs":
        view_text = "View: Borough heatmap"
        agg = d.groupby("Borough").size().reset_index(name="Count")
        fig = px.choropleth_mapbox(agg, geojson=boroughs_geo, locations="Borough",
                                   featureidkey="properties.NAME", color="Count",
                                   hover_name="Borough", hover_data=["Count"], height=800,
                                   mapbox_style="open-street-map", color_continuous_scale=["yellow","orange","red"])
    elif view == "wards":
        view_text = "View: Ward heatmap"
        d2 = d.dropna(subset=["Ward ID", "Ward Name"])

        agg = (
            d2.groupby(["Ward ID", "Ward Name", "Borough"], as_index=False)
            .size()
            .rename(columns={"size": "Count"})
        )
        scale = ["#ffffb2", "#fed976", "#fd8d3c", "#f03b20", "#bd0026"]
        vmax = agg["Count"].quantile(0.95)
        fig = px.choropleth_mapbox(
            agg,
            geojson=wards_geo,
            locations="Ward ID",
            featureidkey="properties.WD24CD",
            color="Count",
            hover_name="Ward Name",
            hover_data=["Count", "Borough"],
            mapbox_style="open-street-map",
            height=800,
            color_continuous_scale=scale,
            range_color=(0, vmax),
            color_continuous_midpoint=vmax / 2
        )
        fig.update_traces(
            marker_line_width=0.5,
            marker_line_color="black",
            selector=dict(type="choroplethmapbox")
        )
    elif view == "officers":
        d_off = pred_df.copy()
        if borough_vals:
            d_off = d_off[d_off["Borough"].isin(borough_vals)]
        if ward_vals:
            d_off = d_off[d_off["Ward Name"].isin(ward_vals)]
        if yols:
            d_off = d_off[d_off["Year"].isin(yols)]
        if moys:
            d_off = d_off[d_off["Month"].isin(moys)]
        if daytype is not None:
            d_off = d_off[d_off["Day Type"] == daytype]
        if timeslot is not None:
            d_off = d_off[d_off["Time Slot"] == timeslot]

        agg_off = (
            d_off
            .groupby(["Ward ID", "Ward Name"], as_index=False)["Officers"]
            .sum()
        )
        fig = px.choropleth_mapbox(
            agg_off,
            geojson=wards_geo,
            locations="Ward ID",
            featureidkey="properties.WD24CD",
            color="Officers",
            hover_name="Ward Name",
            hover_data=["Officers"],
            mapbox_style="carto-positron",
            height=800,
            color_continuous_scale=px.colors.sequential.YlOrRd
        )
        fig.update_traces(
            marker_line_width=0.5,
            marker_line_color="black",
            selector=dict(type="choroplethmapbox")
        )
    else:
        view_text = "View: Crime density"
        fig = px.density_mapbox(d, lat="Latitude", lon="Longitude", radius=10,
                                hover_data=["Month", "Location", "Last outcome category"],
                                zoom=11, height=800, mapbox_style="open-street-map")
    # --- preserve pan & zoom between updates ---
    default_center = {"lon": -0.09, "lat": 51.515}
    default_zoom   = 9
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0]

    if triggered == "reset-button":
        # reset clicked → go home
        lon, lat, zoom = default_center["lon"], default_center["lat"], default_zoom
    else:
        rd = relayout or {}
        center_obj = rd.get("mapbox.center")
        if isinstance(center_obj, dict):
            lon = center_obj.get("lon", default_center["lon"])
            lat = center_obj.get("lat", default_center["lat"])
        else:
            # fallback to flattened keys if present
            lon = rd.get("mapbox.center.lon", default_center["lon"])
            lat = rd.get("mapbox.center.lat", default_center["lat"])

        zoom = rd.get("mapbox.zoom", default_zoom)

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lon": lon, "lat": lat},
        mapbox_zoom=zoom,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Inter"
        )
    )
    return fig, ""
@app.callback(
    [
      Output("location-dropdown",       "value"),
      Output("lsoa-dropdown",           "value"),
      Output("borough-dropdown",        "value"),
      Output("ward-dropdown",           "value"),
      Output("outcome-dropdown",        "value"),
      Output("year-checklist",          "value"),
      Output("month-of-year-checklist", "value"),
      Output("date-picker",             "start_date"),
      Output("date-picker",               "end_date"),
      Output("force-checklist",         "value"),
      Output("view-selector",           "value"),
    ],
    [Input("reset-button", "n_clicks")],
    prevent_initial_call=True
)
def reset_all_filters(n):
    default_forces = df["Reported by"].unique().tolist()
    return (
        [],     # location-dropdown
        [],     # lsoa-dropdown
        [],     # borough-dropdown
        [],     # ward-dropdown
        [],     # outcome-dropdown
        [],     # year-checklist
        [],     # month-of-year-checklist
        "2022-03",  # start-month
        "2025-02",  # end-month
        default_forces,  # force-checklist
        "wards"    # view-selector
    )
# ---- RUN SERVER ----
if __name__ == "__main__":
    app.run(debug=True)
