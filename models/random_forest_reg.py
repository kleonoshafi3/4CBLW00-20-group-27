import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from datetime import datetime
import traceback

def safe_read_csv(file_path):
    """Safely read CSV file with error handling"""
    try:
        logger.info(f"Attempting to read file: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        df_sample = pd.read_csv(file_path, nrows=5)
        logger.info("Sample data structure:\n%s", df_sample.head().to_string())
        logger.info("Columns: %s", df_sample.columns.tolist())
        df = pd.read_csv(file_path, parse_dates=['Month'])
        logger.info(f"Successfully read {len(df)} rows")
        return df
    except Exception:
        logger.exception("Error reading CSV")
        raise

def create_output_dirs():
    """Create directories for saving outputs"""
    try:
        viz_dir = os.path.join("visualizations", "random_forest_reg")
        os.makedirs(viz_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join("models", f"random_forest_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        return viz_dir, model_dir
    except Exception:
        logger.exception("Error creating output directories")
        raise

def create_lag_features(df, lag_periods=[1, 2, 3]):
    df = df.sort_values(['ward_id', 'Year', 'Month_Num'])
    for ward in df['ward_id'].unique():
        mask = df['ward_id'] == ward
        series = df.loc[mask, 'crime_density_per_km2']
        for lag in lag_periods:
            df.loc[mask, f'Crime_Density_Lag_{lag}'] = series.shift(lag)
    return df

def create_rolling_features(df, windows=[3, 6, 12]):
    df = df.sort_values(['ward_id', 'Year', 'Month_Num'])
    for ward in df['ward_id'].unique():
        mask = df['ward_id'] == ward
        series = df.loc[mask, 'crime_density_per_km2']
        for window in windows:
            df.loc[mask, f'Crime_Density_Rolling_{window}'] = (
                series.shift(1).rolling(window=window, min_periods=1).mean()
            )
    return df

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    # 1. Setup output dirs & log file
    viz_dir, model_dir = create_output_dirs()
    log_file = os.path.join(model_dir, 'random_forest.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logger.addHandler(fh)
    logger.info(f"Logging initialized — writing to {log_file}")

    # 2. Load data
    logger.info("Loading data...")
    df = safe_read_csv('data/combined_monthly_imd_data.csv')

    # 3. Prepare data
    df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
    df = df.dropna()
    df = df.drop([c for c in ['crime_count','area_km2'] if c in df.columns], axis=1)
    df['Year'] = df['Month'].dt.year
    df['Month_Num'] = df['Month'].dt.month

    # 4. Feature engineering
    logger.info("Creating lag and rolling features...")
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = df.dropna()
    logger.info(f"Number of rows after feature engineering: {len(df)}")

    # 5. Train/test split by year
    df_train = df[df['Year'] <= 2024].copy()
    df_test  = df[df['Year'] >  2024].copy()
    logger.info("Train covers years <=2024: %d rows", len(df_train))
    logger.info("Test  covers years >2024: %d rows", len(df_test))

    # 6. Encode ward_id
    le = LabelEncoder()
    df_train['ward_id_encoded'] = le.fit_transform(df_train['ward_id'])
    df_test ['ward_id_encoded'] = le.transform(df_test['ward_id'])

    # 7. Define features
    lag_features     = [f'Crime_Density_Lag_{i}'     for i in (1,2,3)]
    rolling_features = [f'Crime_Density_Rolling_{w}'  for w in (3,6,12)]
    weather_features = [
        'Monthly_Weather_Code','Avg_Temp_2m_Min','Avg_Temp_2m_Max',
        'Avg_Wind_Speed_10m_Max','Avg_Daylight_Duration',
        'Avg_Precipitation_Sum','Avg_Precipitation_Hours'
    ]
    imd_features = ['2007', '2010', '2015', '2019', 'pct_change_2007_2019',
                        'rank_2007', 'rank_2010', 'rank_2015', 'rank_2019',
                        'r1', 'r2', 'r3',
                        'mean_annual_rate', 'imd_est_2024', 'rank_est_2024']
    features = (
        ['ward_id_encoded','Year','Month_Num']
      + lag_features + rolling_features
    )
    features.extend(f for f in weather_features + imd_features
                    if f in df_train.columns)
    logger.info("Using features: %s", features)

    X_train, y_train = df_train[features], df_train['crime_density_per_km2']
    X_test,  y_test  = df_test [features], df_test ['crime_density_per_km2']

    # 8. Grid search
    logger.info("Starting grid search for hyperparameters...")
    param_grid = {
        'n_estimators':     [50,100,200],
        'max_depth':        [None,10,20],
        'min_samples_split':[2,5]
    }
    rf = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(
        rf, param_grid, cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    logger.info("Best Parameters: %s", grid.best_params_)

    # 9. Predictions & evaluation
    logger.info("Making predictions on train & test sets…")
    y_train_pred = best_model.predict(X_train)
    y_test_pred  = best_model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae  = mean_absolute_error(y_train, y_train_pred)
    train_r2   = r2_score(y_train, y_train_pred)
    logger.info("TRAIN SET → RMSE: %.3f, MAE: %.3f, R²: %.3f",
                train_rmse, train_mae, train_r2)

    test_rmse  = np.sqrt(mean_squared_error(y_test,  y_test_pred))
    test_mae   = mean_absolute_error(y_test, y_test_pred)
    test_r2    = r2_score(y_test, y_test_pred)
    logger.info(" TEST SET → RMSE: %.3f, MAE: %.3f, R²: %.3f",
                test_rmse, test_mae, test_r2)

    sample_idx = np.random.choice(len(y_test), size=5, replace=False)
    sample_df = pd.DataFrame({
        'ward_id':   df_test.iloc[sample_idx]['ward_id'],
        'Month':     df_test.iloc[sample_idx]['Month'].dt.to_period('M').astype(str),
        'Actual':    y_test.iloc[sample_idx].values,
        'Predicted': y_test_pred[sample_idx]
    })
    logger.info("Sample TEST predictions:\n%s", sample_df.to_string(index=False))

    # 10. Visualizations
    logger.info("Creating and saving visualizations...")
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=y_test, y=y_test_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             color='red', linestyle='--')
    plt.title('Actual vs Predicted - Test Set')
    plt.xlabel('Actual Crime Density')
    plt.ylabel('Predicted Crime Density')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'actual_vs_predicted.png'), dpi=300)
    plt.close()

    importances = best_model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(8,5))
    sns.barplot(data=feat_imp_df, x='Importance', y='Feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'feature_importance.png'), dpi=300)
    plt.close()
    feat_imp_df.to_csv(os.path.join(viz_dir, 'feature_importance.csv'), index=False)

    logger.info("Visualizations have been saved to: %s", viz_dir)

except Exception:
    logger.exception("Fatal error during random forest training")
    raise
