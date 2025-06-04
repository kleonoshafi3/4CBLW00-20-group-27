import pandas as pd

#load two separate weather datasets:
#data comes from https://www.visualcrossing.com/weather-query-builder/london/?v=api#
#it was only possible to obtain a csv for 1000 days at a time, hence two csv files
df1 = pd.read_csv("weather_data_1.csv")
df2 = pd.read_csv("weather_data_2.csv")

#combine them
df = pd.concat([df2, df1])

#see which columns contain missing values and remove those
#if you'd like, this is where to add extra columns to remove more
print(df.isnull().sum())
df = df.drop(['name', 'preciptype', 'severerisk', 'stations', 'sunrise', 'sunset', 'conditions', 'description', 'icon'], axis=1)

#ensure index is datetime
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime', drop=True)
df.index.name = 'Month'

#next, "resample" per month and get averages for everything
monthly_avg = df.resample(rule='MS').mean()
monthly_avg = monthly_avg.round(2)

print(f"This dataset has monthly averages from {monthly_avg.index.min()} to {monthly_avg.index.max()} for attributes {monthly_avg.columns.tolist()}. ")

#creates a separate csv file with only weather data per month
monthly_avg.to_csv('output_csv_files/weather_data.csv')

#adds the weather data to ward_temporal_analysis
df_wards = pd.read_csv("output_csv_files/ward_temporal_analysis.csv")
df_wards['Month'] = pd.to_datetime(df_wards['Month'])
result = pd.merge(df_wards, monthly_avg, on='Month', how='left')
result.to_csv('output_csv_files/ward_temporal_analysis.csv')
