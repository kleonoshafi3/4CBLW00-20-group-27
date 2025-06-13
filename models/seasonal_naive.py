import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def create_output_dirs():
    """Create directories for saving outputs"""
    try:
        # Create visualization directory
        viz_dir = os.path.join("visualizations", "seasonal_naive")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create model directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join("models", f"seasonal_naive_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        return viz_dir, model_dir
    except Exception as e:
        print(f"Error creating output directories: {str(e)}")
        raise

def calculate_seasonal_naive_predictions(df_train, df_test, seasonal_period=12):
    """
    Calculate seasonal naive predictions using the same month from previous year
    """
    # Create a copy of test data
    df_test_pred = df_test.copy()
    
    # For each ward and month in test set
    for ward in df_test['ward_id'].unique():
        ward_mask_test = df_test['ward_id'] == ward
        ward_mask_train = df_train['ward_id'] == ward
        
        for month in range(1, 13):
            # Get the same month from previous year in training data
            month_mask_test = df_test['Month'].dt.month == month
            month_mask_train = df_train['Month'].dt.month == month
            
            # Get the last year's value for this month
            last_year_value = df_train.loc[ward_mask_train & month_mask_train, 'crime_density_per_km2'].iloc[-1]
            
            # Apply this value to all instances of this month in test set
            df_test_pred.loc[ward_mask_test & month_mask_test, 'predicted_density'] = last_year_value
    
    return df_test_pred

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

    # Sort data by time first
    df = df.sort_values(['Month', 'ward_id'])
    
    # Calculate the split point for 70-30 split
    split_idx = int(len(df) * 0.7)
    
    # Split the data temporally
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    print(f"Training set size: {len(df_train)} ({len(df_train)/len(df)*100:.1f}%)")
    print(f"Test set size: {len(df_test)} ({len(df_test)/len(df)*100:.1f}%)")
    
    if len(df_train) == 0 or len(df_test) == 0:
        raise ValueError("Empty training or test set after split")

    # Calculate seasonal naive predictions
    print("\nCalculating seasonal naive predictions...")
    df_test_pred = calculate_seasonal_naive_predictions(df_train, df_test)
    
    # Calculate metrics
    mse = mean_squared_error(df_test['crime_density_per_km2'], df_test_pred['predicted_density'])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(df_test['crime_density_per_km2'], df_test_pred['predicted_density'])
    r2 = r2_score(df_test['crime_density_per_km2'], df_test_pred['predicted_density'])
    
    print("\nModel Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R-squared Score: {r2:.3f}")
    
    # Save metrics to file
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    with open(os.path.join(viz_dir, 'metrics.txt'), 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.3f}\n")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.scatter(df_test['crime_density_per_km2'], df_test_pred['predicted_density'], alpha=0.5)
    plt.plot([df_test['crime_density_per_km2'].min(), df_test['crime_density_per_km2'].max()],
             [df_test['crime_density_per_km2'].min(), df_test['crime_density_per_km2'].max()],
             'r--', lw=2)
    plt.xlabel('Actual Crime Density')
    plt.ylabel('Predicted Crime Density')
    plt.title('Actual vs Predicted Crime Density')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot time series for a few sample wards
    sample_wards = df_test['ward_id'].unique()[:5]  # Take first 5 wards
    plt.figure(figsize=(15, 8))
    
    for ward in sample_wards:
        ward_data = df_test[df_test['ward_id'] == ward]
        ward_pred = df_test_pred[df_test_pred['ward_id'] == ward]
        
        plt.plot(ward_data['Month'], ward_data['crime_density_per_km2'], 
                label=f'Actual - Ward {ward}', alpha=0.7)
        plt.plot(ward_pred['Month'], ward_pred['predicted_density'], 
                '--', label=f'Predicted - Ward {ward}', alpha=0.7)
    
    plt.xlabel('Month')
    plt.ylabel('Crime Density')
    plt.title('Actual vs Predicted Crime Density Over Time (Sample Wards)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'time_series_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot error distribution
    errors = df_test['crime_density_per_km2'] - df_test_pred['predicted_density']
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Errors')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot monthly average errors
    monthly_errors = errors.groupby(df_test['Month'].dt.month).mean()
    plt.figure(figsize=(12, 6))
    monthly_errors.plot(kind='bar')
    plt.xlabel('Month')
    plt.ylabel('Average Error')
    plt.title('Average Prediction Error by Month')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'monthly_errors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations have been saved to: {viz_dir}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Full traceback:")
    print(traceback.format_exc()) 