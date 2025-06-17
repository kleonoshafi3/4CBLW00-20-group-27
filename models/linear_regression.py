import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
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
        viz_dir = os.path.join("visualizations", "linear_reg")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create model directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join("models", f"linear_reg_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        return viz_dir, model_dir
    except Exception as e:
        print(f"Error creating output directories: {str(e)}")
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

    # Define feature creation functions
    # Create rolling means for different windows
    def create_rolling_features(df, windows=[3, 6, 12]):
        """
        Create rolling-mean features for crime density using only past data.
        """
        # Sort so shift/rolling lines up correctly
        df = df.sort_values(['ward_id', 'Year', 'Month_Num'])
        
        for ward in df['ward_id'].unique():
            mask = df['ward_id'] == ward
            series = df.loc[mask, 'crime_density_per_km2']
            for window in windows:
                # shift first, then rolling → only past months included
                rolled = series.shift(1).rolling(window=window, min_periods=1).mean()
                df.loc[mask, f'Crime_Density_Rolling_{window}'] = rolled
        return df


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

    # Create features BEFORE splitting
    print("\nCreating features...")
    df = create_rolling_features(df)
    df = create_lag_features(df)
    
    # Drop rows with NaN values in features
    df = df.dropna()
    print(f"Number of rows after feature engineering: {len(df)}")

    # Sort data by time and split
    df = df.sort_values(['Month', 'ward_id'])
    split_idx = int(len(df) * 0.7)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    print(f"Training set size: {len(df_train)} ({len(df_train)/len(df)*100:.1f}%)")
    print(f"Test set size: {len(df_test)} ({len(df_test)/len(df)*100:.1f}%)")
    
    if len(df_train) == 0 or len(df_test) == 0:
        raise ValueError("Empty training or test set after split")

    # Encode ward_ids after split to prevent leakage
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
                        'r1', 'r2', 'r3',
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

    # Linear Regression
    print("\nStarting model training...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make Predictions
    print("\nMaking predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluation Function
    def evaluate_model(y_true, y_pred, dataset_name=""):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"\n{dataset_name} Evaluation:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R² Score: {r2:.2f}")
        return rmse, mae, r2

    # Evaluate Model
    train_metrics = evaluate_model(y_train, y_train_pred, "Training Set")
    test_metrics = evaluate_model(y_test, y_test_pred, "Test Set")

    # Save evaluation metrics
    metrics_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'R² Score'] * 2,
        'Value': [train_metrics[0], train_metrics[1], train_metrics[2],
                  test_metrics[0], test_metrics[1], test_metrics[2]],
        'Dataset': ['Training'] * 3 + ['Test'] * 3
    })
    metrics_df.to_csv(os.path.join(viz_dir, 'evaluation_metrics.csv'), index=False)

    # Create and save visualizations
    print("\nCreating and saving visualizations...")

    # 1. Actual vs Predicted (Test Set)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.scatterplot(x=y_test, y=y_test_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.title('Actual vs Predicted - Test Set')
    plt.xlabel('Actual Crime Density')
    plt.ylabel('Predicted Crime Density')
    plt.grid(True)

    # 2. Feature Importance (Coefficients) - Improved visualization
    plt.subplot(1, 3, 2)
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    })
    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
    
    # Create horizontal bar plot with better formatting
    plt.barh(feature_importance['Feature'], feature_importance['Coefficient'])
    plt.title('Linear Regression Coefficients', fontsize=14)
    plt.xlabel('Coefficient', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 3. Metrics Comparison
    plt.subplot(1, 3, 3)
    sns.barplot(data=metrics_df, x='Metric', y='Value', hue='Dataset', palette='Set2')
    plt.title('Model Performance: Training vs Test Set')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(os.path.join(viz_dir, 'model_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save feature importance
    feature_importance.to_csv(os.path.join(viz_dir, 'feature_importance.csv'), index=False)

    # Correlation Matrix Heatmap
    print("\nCreating correlation matrix heatmap...")
    correlation_matrix = df_train[features + ['crime_density_per_km2']].corr()
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(correlation_matrix, 
                annot=True,
                cmap='RdBu_r',
                center=0,
                fmt='.2f',
                square=True,
                cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
                annot_kws={'size': 10})
    
    plt.title('Feature Correlation Heatmap', fontsize=24, pad=20)
    plt.xticks(fontsize=12, rotation=90, ha='center')
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Residuals Analysis
    print("\nPerforming residuals analysis...")
    
    # Calculate residuals
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred

    # Residuals vs Predicted
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test_pred, test_residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs Predicted Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.savefig(os.path.join(viz_dir, 'residuals_vs_predicted.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Residuals Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(test_residuals, kde=True)
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(viz_dir, 'residuals_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save the model
    print("\nSaving model...")
    import joblib
    joblib.dump(model, os.path.join(model_dir, 'linear_regression_model.joblib'))
    print(f"Model saved as '{os.path.join(model_dir, 'linear_regression_model.joblib')}'")

    # Save model parameters
    with open(os.path.join(model_dir, 'model_parameters.txt'), 'w') as f:
        f.write(f"Model Type: Linear Regression\n")
        f.write(f"Features used: {features}\n")
        f.write(f"Training set size: {len(X_train)}\n")
        f.write(f"Test set size: {len(X_test)}\n")
        f.write(f"Model coefficients:\n")
        for feature, coef in zip(features, model.coef_):
            f.write(f"{feature}: {coef:.6f}\n")

    print(f"\nAll visualizations have been saved to: {viz_dir}")
    print(f"Model and parameters have been saved to: {model_dir}")

except Exception as e:
    print(f"\nError occurred: {str(e)}")
    print("\nFull traceback:")
    print(traceback.format_exc())
    raise 