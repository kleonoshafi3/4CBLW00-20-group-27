import pandas as pd
from pathlib import Path
import dash
from dash import dcc, html, Input, Output
from dash import dash_table
import plotly.express as px

# ---- 1. LOAD & PREPARE ----
def load_and_prepare(data_dir: str):
    data_path = Path(data_dir)
    frames = []
    for month_dir in sorted(data_path.iterdir()):
        if not month_dir.is_dir():
            continue
        for force in ("metropolitan", "city-of-london"):
            fp = month_dir / f"{month_dir.name}-{force}-street.csv"
            if fp.exists():
                df = pd.read_csv(fp)
                df["Reported by"] = df["Reported by"].fillna(force.title())
                frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full["MonthDt"] = pd.to_datetime(full["Month"], format="%Y-%m", errors="coerce")
    full = full[full["Crime type"].str.strip().str.lower() == "burglary"].copy()
    return full

# ---- MAIN DATAFRAME ----
DATA_DIR = Path(__file__).parent / "data"
df = load_and_prepare(str(DATA_DIR))
assert df["Crime type"].str.strip().str.lower().eq("burglary").all(), "Non-burglary records found!"
if df.empty:
    raise ValueError("❗ No burglary records found.")

# compute slider bounds
years = sorted(df["MonthDt"].dt.year.dropna().astype(int).unique().tolist())
months = sorted(df["MonthDt"].dt.month.dropna().astype(int).unique().tolist())
year_marks = {y: str(y) for y in years}
month_marks = {m: str(m) for m in months}

# ---- DASH LAYOUT ----
app = dash.Dash(__name__, title="London Burglary Dashboard")
app.layout = html.Div([
    # Header and counter
    html.Div([
        html.H1("London Burglary Dashboard", style={"textAlign": "center"}),
        html.H3(id="count-div", style={"textAlign": "center"})
    ]),

    html.Div([
        html.Div([
            html.H2("Filter Burglaries", style={"margin-bottom": "10px"}),

            html.Label("Select Location(s):"),
            dcc.Dropdown(
                id="location-dropdown",
                options=[{"label": loc, "value": loc} for loc in sorted(df["Location"].dropna().unique())],
                value=[], multi=True, placeholder="Select locations...",
                style={"width": "100%"}
            ), html.Br(),

            dash_table.DataTable(
                id="location-table",
                columns=[{"name": "Location", "id": "Location"}, {"name": "Count", "id": "Count"}],
                data=[], style_table={"height": "200px", "overflowY": "auto"}
            ), html.Br(),

            html.Label("Select LSOA area(s):"),
            dcc.Dropdown(
                id="lsoa-dropdown",
                options=[{"label": lsoa, "value": lsoa} for lsoa in sorted(df["LSOA name"].dropna().unique())],
                value=[], multi=True, placeholder="Select LSOAs...",
                style={"width": "100%"}
            ), html.Br(),

            dash_table.DataTable(
                id="lsoa-table",
                columns=[{"name": "LSOA name", "id": "LSOA name"}, {"name": "Count", "id": "Count"}],
                data=[], style_table={"height": "200px", "overflowY": "auto"}
            ), html.Br(),

            html.Label("Year Range:"),
            dcc.RangeSlider(
                id="year-slider", min=years[0], max=years[-1],
                value=[years[0], years[-1]], marks=year_marks, step=1
            ), html.Br(),

            html.Label("Month Range:"),
            dcc.RangeSlider(
                id="month-slider", min=months[0], max=months[-1],
                value=[months[0], months[-1]], marks=month_marks, step=1
            ), html.Br(),

            html.Label("Police Force:"),
            dcc.Checklist(
                id="force-checklist",
                options=[{"label": f, "value": f} for f in df["Reported by"].unique()],
                value=df["Reported by"].unique().tolist(), inline=True
            ), html.Br(),

            html.Label("Outcome Category:"),
            dcc.Checklist(
                id="outcome-checklist",
                options=[{"label": o, "value": o} for o in sorted(df["Last outcome category"].dropna().unique())],
                value=sorted(df["Last outcome category"].dropna().unique()), inline=False
            ), html.Br(),

            html.Div([
                html.Button("Toggle View", id="toggle-btn", n_clicks=0,
                            style={"background-color": "#0074D9", "color": "white"}),
                html.Button("Reset Filters", id="reset-btn", n_clicks=0,
                            style={"background-color": "#FF4136", "color": "white", "marginLeft": "10px"})
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div(id="view-label", style={"marginTop": "5px", "fontStyle": "italic"}),

        ], style={"position": "fixed", "top": "100px", "left": "0", "width": "300px", "bottom": "0", "padding": "20px", "background-color": "#f4f4f4", "overflow": "auto"}),

        html.Div([dcc.Graph(id="map-graph", style={"height": "100vh"})], style={"margin-left": "320px"})
    ])
])

# ---- CALLBACK: populate tables ----
@app.callback(
    [Output("location-table", "data"), Output("lsoa-table", "data")],
    [Input("location-dropdown", "value"), Input("lsoa-dropdown", "value"),
     Input("year-slider", "value"), Input("month-slider", "value"),
     Input("force-checklist", "value"), Input("outcome-checklist", "value"),
     Input("toggle-btn", "n_clicks")]
)
def update_tables(loc_vals, lsoa_vals, year_range, month_range, forces, outcomes, toggle):
    # apply time and checklist filters first
    base = df[
        df["MonthDt"].dt.year.between(year_range[0], year_range[1]) &
        df["MonthDt"].dt.month.between(month_range[0], month_range[1]) &
        df["Reported by"].isin(forces) &
        df["Last outcome category"].isin(outcomes)
    ]
    # determine valid LSOAs (intersection) when locations selected
    if loc_vals:
        subset = base[base["Location"].isin(loc_vals)]
        lsoa_counts = subset.groupby("LSOA name")["Location"].nunique()
        valid_lsoas = lsoa_counts[lsoa_counts == len(loc_vals)].index.tolist()
    else:
        valid_lsoas = base["LSOA name"].dropna().unique().tolist()
    # determine valid Locations when LSOAs selected
    if lsoa_vals:
        subset = base[base["LSOA name"].isin(lsoa_vals)]
        loc_counts = subset.groupby("Location")["LSOA name"].nunique()
        valid_locs = loc_counts[loc_counts == len(lsoa_vals)].index.tolist()
    else:
        valid_locs = base["Location"].dropna().unique().tolist()
    # filter for table data
    loc_df = base[base["Location"].isin(valid_locs)]
    loc_df = loc_df.groupby("Location").size().reset_index(name="Count").sort_values(by="Count", ascending=False)
    lsoa_df = base[base["LSOA name"].isin(valid_lsoas)]
    lsoa_df = lsoa_df.groupby("LSOA name").size().reset_index(name="Count").sort_values(by="Count", ascending=False)
    return loc_df.to_dict('records'), lsoa_df.to_dict('records')

# ---- CALLBACK: dynamic LSOA options based on other filters ----
@app.callback(
    Output("lsoa-dropdown", "options"),
    [Input("location-dropdown", "value"), Input("year-slider", "value"),
     Input("month-slider", "value"), Input("force-checklist", "value"),
     Input("outcome-checklist", "value")]
)
def update_lsoa_options(selected_locs, year_range, month_range, forces, outcomes):
    d = df[
        df["MonthDt"].dt.year.between(year_range[0], year_range[1]) &
        df["MonthDt"].dt.month.between(month_range[0], month_range[1]) &
        df["Reported by"].isin(forces) &
        df["Last outcome category"].isin(outcomes)
    ]
    if selected_locs:
        subset = d[d["Location"].isin(selected_locs)]
        counts = subset.groupby("LSOA name")["Location"].nunique()
        valid = counts[counts == len(selected_locs)].index.tolist()
    else:
        valid = d["LSOA name"].dropna().unique().tolist()
    return [{"label": lsoa, "value": lsoa} for lsoa in sorted(valid)]

# ---- CALLBACK: dynamic Location options based on other filters ----
@app.callback(
    Output("location-dropdown", "options"),
    [Input("lsoa-dropdown", "value"), Input("year-slider", "value"),
     Input("month-slider", "value"), Input("force-checklist", "value"),
     Input("outcome-checklist", "value")]
)
def update_location_options(selected_lsoas, year_range, month_range, forces, outcomes):
    d = df[
        df["MonthDt"].dt.year.between(year_range[0], year_range[1]) &
        df["MonthDt"].dt.month.between(month_range[0], month_range[1]) &
        df["Reported by"].isin(forces) &
        df["Last outcome category"].isin(outcomes)
    ]
    if selected_lsoas:
        subset = d[d["LSOA name"].isin(selected_lsoas)]
        counts = subset.groupby("Location")["LSOA name"].nunique()
        valid = counts[counts == len(selected_lsoas)].index.tolist()
    else:
        valid = d["Location"].dropna().unique().tolist()
    return [{"label": loc, "value": loc} for loc in sorted(valid)]

# ---- CALLBACK: reset filters ----
@app.callback(
    [Output("location-dropdown", "value"), Output("lsoa-dropdown", "value"),
     Output("year-slider", "value"), Output("month-slider", "value"),
     Output("force-checklist", "value"), Output("outcome-checklist", "value")],
    Input("reset-btn", "n_clicks"),
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    return (
        [], [], [years[0], years[-1]], [months[0], months[-1]],
        df["Reported by"].unique().tolist(),
        sorted(df["Last outcome category"].dropna().unique())
    )

# ---- CALLBACK: update map ----
@app.callback(
    [Output("view-label", "children"), Output("count-div", "children"), Output("map-graph", "figure")],
    [Input("location-dropdown", "value"), Input("lsoa-dropdown", "value"),
     Input("year-slider", "value"), Input("month-slider", "value"),
     Input("force-checklist", "value"), Input("outcome-checklist", "value"),
     Input("toggle-btn", "n_clicks")]
)
def update_dashboard(loc_vals, lsoa_vals, year_range, month_range, forces, outcomes, toggle):
    d = df.copy()
    if loc_vals:
        d = d[d["Location"].isin(loc_vals)]
    if lsoa_vals:
        d = d[d["LSOA name"].isin(lsoa_vals)]
    d = d[
        d["MonthDt"].dt.year.between(year_range[0], year_range[1]) &
        d["MonthDt"].dt.month.between(month_range[0], month_range[1]) &
        d["Reported by"].isin(forces) &
        d["Last outcome category"].isin(outcomes)
    ]
    count_text = f"Total burglaries: {len(d):,}"
    view_text = "View: " + ("Points" if toggle % 2 else "Aggregate")
    if toggle % 2 == 0:
        agg2 = (
            d.groupby(["LSOA code", "LSOA name"]).size()
             .reset_index(name="Count")
             .merge(
                 d[["LSOA code", "Latitude", "Longitude"]].drop_duplicates(subset=["LSOA code"]), on="LSOA code"
             )
        )
        fig = px.scatter_mapbox(
            agg2, lat="Latitude", lon="Longitude",
            size="Count", color="Count",
            hover_name="LSOA name", hover_data=["Count"],
            size_max=30, zoom=12, height=800,
            color_continuous_scale=px.colors.diverging.RdBu[::-1]
        )
    else:
        fig = px.scatter_mapbox(
            d, lat="Latitude", lon="Longitude",
            hover_name="Crime ID",
            hover_data=["Month", "Reported by", "Last outcome category", "Location"],
            zoom=12, height=800, opacity=0.6
        )
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": 51.515, "lon": -0.09},
        margin={"l": 0, "r": 0, "t": 0, "b": 0}
    )
    return view_text, count_text, fig

# ---- RUN SERVER ----
if __name__ == "__main__":
    app.run_server(debug=True)
