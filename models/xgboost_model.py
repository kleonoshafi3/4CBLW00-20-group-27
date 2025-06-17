import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import TimeSeriesSplit
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
        viz_dir = os.path.join("visualizations", "xgboost")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create model directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join("models", f"model_{timestamp}")
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
    df['month_sin'] = np.sin(2 * np.pi * df['Month_Num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month_Num'] / 12)
    print(f"Year range: {df['Year'].min()} to {df['Year'].max()}")

    # Create output directories
    viz_dir, model_dir = create_output_dirs()
    print(f"Visualizations will be saved to: {viz_dir}")
    print(f"Model will be saved to: {model_dir}")

    # 1) Get the sorted list of unique months
    months = np.sort(df['Month'].unique())

    # 2) Pick the 70%‐ile month
    cutoff_month = months[int(len(months) * 0.7)]

    # 3) Split so that train = months ≤ cutoff, test = months > cutoff
    df_train = df[df['Month'] <= cutoff_month].copy()
    df_test  = df[df['Month']  > cutoff_month].copy()

    print("Train covers up to:", cutoff_month)
    print("Test starts from:", df_test['Month'].min())

    
    print(f"Training set size: {len(df_train)} ({len(df_train)/len(df)*100:.1f}%)")
    print(f"Test set size: {len(df_test)} ({len(df_test)/len(df)*100:.1f}%)")
    
    if len(df_train) == 0 or len(df_test) == 0:
        raise ValueError("Empty training or test set after split")

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


    # Create lag features
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

    # Create features for training and test sets separately
    print("\nCreating features...")
    
    # For training set, we can use all features
    df_train = create_rolling_features(df_train)
    df_train = create_lag_features(df_train)
    
    # For test set, we need to handle the first few rows differently
    df_test = create_rolling_features(df_test)
    df_test = create_lag_features(df_test)

    # Add rolling-momentum: 3-month vs 6-month rolling mean difference
    for d in (df_train, df_test):
        d['Rolling_Momentum_3_6'] = (
            d['Crime_Density_Rolling_3']
            - d['Crime_Density_Rolling_6']
        )
    
    # Drop rows with NaN values in features
    df_train = df_train.dropna()
    df_test = df_test.dropna()
    
    print(f"Training set size after feature engineering: {len(df_train)}")
    print(f"Test set size after feature engineering: {len(df_test)}")

    # Manual target‐encode ward_id by its historical mean crime density
    ward_means = df_train.groupby('ward_id')['crime_density_per_km2'].mean()
    df_train['ward_te'] = df_train['ward_id'].map(ward_means)
    df_test ['ward_te'] = df_test ['ward_id'].map(ward_means)

    # Define features
    lag_features = [f'Crime_Density_Lag_{lag}' for lag in [1, 2, 3]]
    rolling_features = [f'Crime_Density_Rolling_{window}' for window in [3, 6, 12]]
    # weather_features = ['Monthly_Weather_Code', 'Avg_Temp_2m_Min', 'Avg_Temp_2m_Max',
    #                    'Avg_Wind_Speed_10m_Max', 'Avg_Daylight_Duration',
    #                    'Avg_Precipitation_Sum', 'Avg_Precipitation_Hours']
    # imd_features = ['2007', '2010', '2015', '2019', 'pct_change_2007_2019',
    #                 'rank_2007', 'rank_2010', 'rank_2015', 'rank_2019',
    #                 'predicted_rank_2024', 'predicted_rank_2024_int', 'r1', 'r2', 'r3',
    #                 'mean_annual_rate', 'imd_est_2024', 'rank_est_2024']
    weather_features = ['Monthly_Weather_Code', 'Avg_Temp_2m_Min', 'Avg_Temp_2m_Max',
                        'Avg_Wind_Speed_10m_Max', 'Avg_Daylight_Duration',
                        'Avg_Precipitation_Sum', 'Avg_Precipitation_Hours']
    imd_features = ['2007', '2010', '2015', '2019', 'pct_change_2007_2019',
                        'rank_2007', 'rank_2010', 'rank_2015', 'rank_2019',
                        'r1', 'r2', 'r3',
                        'mean_annual_rate', 'imd_est_2024', 'rank_est_2024']
    momentum_features = ['Rolling_Momentum_3_6']
    cyclical_month   = ['month_sin', 'month_cos']

    # Filter features to only include those that exist in the dataset
    available_features = df_train.columns.tolist()
    features = ['ward_te', 'Year'] + lag_features + rolling_features + cyclical_month + momentum_features
    features.extend([f for f in weather_features + imd_features if f in available_features])

    print("Using features:", features)

    X_train = df_train[features]
    y_train = df_train['crime_density_per_km2']
    X_test = df_test[features]
    y_test = df_test['crime_density_per_km2']

    # XGBoost + Grid Search with more conservative parameters
    print("\nStarting model training...")
    param_grid = {
        'n_estimators':     [100, 110, 120],
        'max_depth':        [4, 5, 6],
        'learning_rate':    [0.11, 0.12, 0.13],
        'subsample':        [0.6, 0.7],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'min_child_weight': [2, 3, 4],
        'gamma':            [0, 0.2],
        'reg_alpha':        [0, 0.1],
        'reg_lambda':       [1, 2]
    }

    xgb_model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        early_stopping_rounds=10  # Added early stopping
    )

    grid = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    # Add validation set for early stopping
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    grid.fit(
        X_train_fit, y_train_fit,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    best_model = grid.best_estimator_
    print(f"Best Parameters: {grid.best_params_}")

    # Make Predictions
    print("\nMaking predictions...")
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # ——————————————————————————————————————————————
    # Build a CSV of ward / year_month → predicted density
    df_pred = df_test[['ward_id', 'Month']].copy()
    # convert Month to “YYYY-MM” string
    df_pred['year_month'] = df_pred['Month'].dt.to_period('M').astype(str)
    # attach your predictions
    df_pred['predicted_density'] = y_test_pred
    # select & order columns exactly as requested
    df_pred = df_pred[['ward_id', 'year_month', 'predicted_density']]

    # save to CSV alongside your model outputs
    pred_csv = os.path.join(model_dir, 'monthly_predicted_crime_density.csv')
    df_pred.to_csv(pred_csv, index=False)
    print(f"Predictions written to {pred_csv}")
    # ——————————————————————————————————————————————

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

    # 2. Feature Importance
    plt.subplot(1, 3, 2)
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    sns.barplot(data=feature_importance, x='Importance', y='Feature')
    plt.title('Feature Importance')
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

    # SHAP Analysis
    print("\nPerforming SHAP Analysis...")
    try:
        # Create a SHAP explainer using the best model
        explainer = shap.TreeExplainer(best_model)

        # Calculate SHAP values for a subset of the test set to avoid memory issues
        sample_size = min(1000, len(X_test))
        X_test_sample = X_test.sample(n=sample_size, random_state=42)
        shap_values = explainer.shap_values(X_test_sample)

        # Create a more readable feature names dictionary
        feature_names = {col: col.replace('_', ' ').title() for col in features}

        # Rename features in X_test_sample for better visualization
        X_test_sample_renamed = X_test_sample.copy()
        X_test_sample_renamed.columns = [feature_names[col] for col in X_test_sample.columns]

        # Plot and save SHAP summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_sample_renamed, plot_type="bar", show=False)
        plt.title("Feature Importance (SHAP Values)", fontsize=14, pad=20)
        plt.xlabel("mean |SHAP value| (impact on model output)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Plot and save detailed SHAP values
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_sample_renamed, show=False)
        plt.title("SHAP Value Distribution", fontsize=14, pad=20)
        plt.xlabel("SHAP value (impact on model output)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'shap_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Plot and save SHAP dependence plots for top features
        top_features = feature_importance.nlargest(3, 'Importance')['Feature'].tolist()
        for feat in top_features:
            plt.figure(figsize=(10, 6))
            # map raw feature to its "pretty" column name
            pretty = feature_names[feat]  
            # call dependence_plot with the pretty name (must match X_test_sample_renamed.columns)
            shap.dependence_plot(pretty, shap_values, X_test_sample_renamed, show=False)
            plt.title(f"SHAP Dependence Plot for {pretty}", fontsize=14, pad=20)
            plt.tight_layout()
            # sanitize filename
            safe = pretty.replace(' ', '_')
            plt.savefig(os.path.join(viz_dir, f'shap_dependence_{safe}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"Warning: SHAP analysis encountered an error: {str(e)}")
        print("Continuing with the rest of the analysis...")

    # Save the model
    print("\nSaving model...")
    import joblib
    joblib.dump(best_model, os.path.join(model_dir, 'xgboost_model.joblib'))
    print(f"Model saved as '{os.path.join(model_dir, 'xgboost_model.joblib')}'")

    # Save model parameters
    with open(os.path.join(model_dir, 'model_parameters.txt'), 'w') as f:
        f.write(f"Best Parameters: {grid.best_params_}\n")
        f.write(f"Features used: {features}\n")
        f.write(f"Training set size: {len(X_train)}\n")
        f.write(f"Test set size: {len(X_test)}\n")

    print(f"\nAll visualizations have been saved to: {viz_dir}")
    print(f"Model and parameters have been saved to: {model_dir}")

except Exception as e:
    print(f"\nError occurred: {str(e)}")
    print("\nFull traceback:")
    print(traceback.format_exc())
    raise 