import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.graphics.tsaplots import plot_acf

# Create visualization directory if it doesn't exist
viz_dir = os.path.join("visualizations")
os.makedirs(viz_dir, exist_ok=True)

# Get the absolute path to the file
file_path = os.path.join("data", "combined_monthly_imd_data.csv")

# Load and prepare data
print("Loading data...")
df = pd.read_csv(file_path, parse_dates=['Month'])
df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
df = df.dropna()

# Extract year and numeric month
df['Year'] = df['Month'].dt.year
df['Month_Num'] = df['Month'].dt.month

print("Creating ACF plot...")
# ACF Plot
monthly_series = df.set_index('Month').resample('ME')['crime_density_per_km2'].mean()

plt.figure(figsize=(12, 6))
plot_acf(monthly_series.dropna(), lags=24)
plt.title('Autocorrelation of Monthly Crime Density', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'acf_plot.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Creating monthly average plot...")
# Monthly Average Plot
df['Month_Num'] = df['Month'].dt.month
monthly_avg = df.groupby('Month_Num')['crime_density_per_km2'].mean()

plt.figure(figsize=(12, 6))
monthly_avg.plot(kind='bar')
plt.title('Average Crime Density by Month', fontsize=20)
plt.xlabel('Month', fontsize=16)
plt.ylabel('Average Crime Density', fontsize=16)
plt.grid(True)
plt.xticks(ticks=range(12), labels=[
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'monthly_average.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Creating seasonal analysis plot...")
# Seasonal Analysis
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

# Add season column
df['Season'] = df['Month'].dt.month.apply(get_season)

# Create seasonal analysis plot
plt.figure(figsize=(12, 6))

# Plot each season with different colors
seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
colors = ['blue', 'green', 'red', 'orange']

for season, color in zip(seasons, colors):
    season_data = df[df['Season'] == season]
    plt.scatter(season_data['Month'], season_data['crime_density_per_km2'], 
                label=season, color=color, alpha=0.6)

# Add trend line
z = np.polyfit(range(len(df)), df['crime_density_per_km2'], 1)
p = np.poly1d(z)
plt.plot(df['Month'], p(range(len(df))), 'w--', label='Trend Line')

# Titles and labels
plt.title('Seasonal Crime Density Patterns in London', fontsize=20)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Crime Density', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'seasonal_patterns.png'), dpi=300, bbox_inches='tight')
plt.close()

# Calculate and save average crime density by season
seasonal_avg = df.groupby('Season')['crime_density_per_km2'].mean().reindex(seasons)
seasonal_avg.to_csv(os.path.join(viz_dir, 'seasonal_averages.csv'))

print(f"All visualizations have been saved to: {viz_dir}") 