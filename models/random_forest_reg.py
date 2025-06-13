import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from datetime import datetime
import traceback

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import shap
from datetime import datetime
import traceback

def safe_read_csv(file_path):
    """Safely read CSV file with error handling"""
    try:
        print(f"Attempting to read file: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read first few rows to check structure
        df_sample = pd.read_csv(file_path, nrows=5)
        print("Sample data structure:")
        print(df_sample.head())
        print("\nColumns:", df_sample.columns.tolist())
        
        # Read full file
        df = pd.read_csv(file_path, parse_dates=['Month'])
        print(f"Successfully read {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        raise

# Create directories for outputs
def create_output_dirs():
    """Create directories for saving outputs"""
    try:
        # Create visualization directory
        viz_dir = os.path.join("visualizations", "random_forest_reg")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create model directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join("models", f"random_forest_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        return viz_dir, model_dir
    except Exception as e:
        print(f"Error creating output directories: {str(e)}")
        raise

def create_lag_features(df, lag_periods=[1, 2, 3]):
    """
    Create lag features for crime density
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe with time series data
    lag_periods : list
        List of lag periods to create
    Returns:
    --------
    pandas DataFrame with added lag features
    """
    try:
        # Sort the dataframe by ward_id, Year, and Month_Num
        df = df.sort_values(['ward_id', 'Year', 'Month_Num'])
        
        # Create lag features for each ward separately
        for ward in df['ward_id'].unique():
            ward_mask = df['ward_id'] == ward
            for lag in lag_periods:
                df.loc[ward_mask, f'Crime_Density_Lag_{lag}'] = df.loc[ward_mask, 'crime_density_per_km2'].shift(lag)
        
        return df
    except Exception as e:
        print(f"Error in create_lag_features: {str(e)}")
        print("DataFrame info:")
        print(df.info())
        raise

def create_rolling_features(df, windows=[3, 6, 12]):
    """
    Create rolling window features for crime density
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe with time series data
    windows : list
        List of window sizes for rolling features
    Returns:
    --------
    pandas DataFrame with added rolling features
    """
    try:
        # Sort the dataframe by ward_id, Year, and Month_Num
        df = df.sort_values(['ward_id', 'Year', 'Month_Num'])
        
        # Create rolling features for each ward separately
        for ward in df['ward_id'].unique():
            ward_mask = df['ward_id'] == ward
            for window in windows:
                df.loc[ward_mask, f'Crime_Density_Rolling_{window}'] = (
                    df.loc[ward_mask, 'crime_density_per_km2']
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
        
        return df
    except Exception as e:
        print(f"Error in create_rolling_features: {str(e)}")
        print("DataFrame info:")
        print(df.info())
        raise

try:
    # Load and prepare data
    print("Loading data...")
    df = safe_read_csv('data/combined_monthly_imd_data.csv')
    print("Data loaded successfully. Columns:", df.columns.tolist())

    # Verify column names
    required_columns = ['ward_id', 'Month', 'crime_density_per_km2']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the dataset. Available columns: {df.columns.tolist()}")

    df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
    print(f"Number of rows after datetime conversion: {len(df)}")
    
    df = df.dropna()
    print(f"Number of rows after dropping NA: {len(df)}")

    # Drop multicollinear variables if they exist
    columns_to_drop = ['crime_count', 'area_km2']
    df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)
    print("Columns after dropping multicollinear variables:", df.columns.tolist())

    # Extract year and numeric month
    df['Year'] = df['Month'].dt.year
    df['Month_Num'] = df['Month'].dt.month
    print(f"Year range: {df['Year'].min()} to {df['Year'].max()}")

    # Create output directories
    viz_dir, model_dir = create_output_dirs()
    print(f"Visualizations will be saved to: {viz_dir}")
    print(f"Model will be saved to: {model_dir}")

    # Create features
    print("\nCreating features...")
    df = create_lag_features(df)
    df = create_rolling_features(df)
    print(f"Number of rows after feature engineering: {len(df)}")

    # Train-test split
    df_train = df[df['Year'] <= 2024]
    df_test = df[df['Year'] > 2024]

    # Encode ward_id
    le = LabelEncoder()
    df_train['ward_id_encoded'] = le.fit_transform(df_train['ward_id'])
    df_test['ward_id_encoded'] = le.transform(df_test['ward_id'])

    # Define features
    lag_features = [f'Crime_Density_Lag_{lag}' for lag in [1, 2, 3]]
    rolling_features = [f'Crime_Density_Rolling_{window}' for window in [3, 6, 12]]
    weather_features = ['Monthly_Weather_Code', 'Avg_Temp_2m_Min', 'Avg_Temp_2m_Max', 
                       'Avg_Wind_Speed_10m_Max', 'Avg_Daylight_Duration', 
                       'Avg_Precipitation_Sum', 'Avg_Precipitation_Hours']
    imd_features = ['2007', '2010', '2015', '2019', 'pct_change_2007_2019', 
                    'rank_2007', 'rank_2010', 'rank_2015', 'rank_2019',
                    'predicted_rank_2024', 'predicted_rank_2024_int', 'r1', 'r2', 'r3',
                    'mean_annual_rate', 'imd_est_2024', 'rank_est_2024']

    # Filter features to only include those that exist in the dataset
    available_features = df_train.columns.tolist()
    features = ['ward_id_encoded', 'Year', 'Month_Num'] + lag_features + rolling_features
    features.extend([f for f in weather_features + imd_features if f in available_features])

    print("Using features:", features)

    X_train = df_train[features]
    y_train = df_train['crime_density_per_km2']
    X_test = df_test[features]
    y_test = df_test['crime_density_per_km2']

    # Grid Search for Hyperparameters
    print("\nStarting grid search for hyperparameters...")
    param_grid = {
        'n_estimators': [50,100,200],  # Reduced from [50, 100, 200]
        'max_depth': [None,10,20],    # Reduced from [None, 10, 20]
        'min_samples_split': [2,5]  # Reduced from [2, 5]
    }

    print("Initializing Random Forest model...")
    rf_model = RandomForestRegressor(random_state=42)
    
    print("Setting up GridSearchCV...")
    grid = GridSearchCV(
        rf_model,
        param_grid,
        cv=3,  # Reduced from 5 folds
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2  # Increased verbosity
    )
    
    print("Starting grid search...")
    grid.fit(X_train, y_train)
    print("Grid search completed!")

    best_model = grid.best_estimator_
    print(f"Best Parameters: {grid.best_params_}")

    # Make Predictions
    print("\nMaking predictions...")
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    print("Predictions completed!")

    # Create and save visualizations
    print("\nCreating and saving visualizations...")

    # 1. Actual vs Predicted (Test Set)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_test_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.title('Actual vs Predicted - Test Set')
    plt.xlabel('Actual Crime Density')
    plt.ylabel('Predicted Crime Density')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Feature Importance
    importances = best_model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=feat_imp_df, x='Importance', y='Feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save feature importance to CSV
    feat_imp_df.to_csv(os.path.join(viz_dir, 'feature_importance.csv'), index=False)

    print(f"\nVisualizations have been saved to: {viz_dir}")

except Exception as e:
    print(f"\nError occurred: {str(e)}")
    print("\nFull traceback:")
    print(traceback.format_exc())
    raise 