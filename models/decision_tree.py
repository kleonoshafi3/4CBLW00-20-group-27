import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
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
        viz_dir = os.path.join("visualizations", "decision_tree")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create model directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join("models", f"decision_tree_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        return viz_dir, model_dir
    except Exception as e:
        print(f"Error creating output directories: {str(e)}")
        raise

def create_rolling_features(df, windows=[3, 6, 12]):
    """Create rolling mean features for crime density"""
    try:
        # Sort the dataframe by ward_id, Year, and Month_Num
        df = df.sort_values(['ward_id', 'Year', 'Month_Num'])
        
        # Create rolling means for each ward separately
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

def create_lag_features(df, lag_periods=[1, 2, 3]):
    """Create lag features for crime density"""
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

    # Create features for training and test sets separately
    print("\nCreating features...")
    
    # For training set, we can use all features
    df_train = create_rolling_features(df_train)
    df_train = create_lag_features(df_train)
    
    # For test set, we need to handle the first few rows differently
    df_test = create_rolling_features(df_test)
    df_test = create_lag_features(df_test)
    
    # Drop rows with NaN values in features
    df_train = df_train.dropna()
    df_test = df_test.dropna()
    
    print(f"Training set size after feature engineering: {len(df_train)}")
    print(f"Test set size after feature engineering: {len(df_test)}")

    # Create target variable using quantiles
    print("\nCreating target variable...")
    # Calculate quantiles for crime density
    q1 = df_train['crime_density_per_km2'].quantile(0.33)
    q2 = df_train['crime_density_per_km2'].quantile(0.67)
    
    # Create three categories based on quantiles
    df_train['crime_level'] = pd.cut(df_train['crime_density_per_km2'], 
                                   bins=[-float('inf'), q1, q2, float('inf')],
                                   labels=[0, 1, 2])
    
    # Apply same thresholds to test set
    df_test['crime_level'] = pd.cut(df_test['crime_density_per_km2'],
                                  bins=[-float('inf'), q1, q2, float('inf')],
                                  labels=[0, 1, 2])
    
    print("Crime level distribution in training set:")
    print(df_train['crime_level'].value_counts(normalize=True))
    print("\nCrime level distribution in test set:")
    print(df_test['crime_level'].value_counts(normalize=True))

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
                    'predicted_rank_2024', 'predicted_rank_2024_int', 'r1', 'r2', 'r3',
                    'mean_annual_rate', 'imd_est_2024', 'rank_est_2024']

    # Filter features to only include those that exist in the dataset
    available_features = df_train.columns.tolist()
    features = ['ward_id_encoded', 'Year', 'Month_Num'] + lag_features + rolling_features
    features.extend([f for f in weather_features + imd_features if f in available_features])

    print("Using features:", features)

    X_train = df_train[features]
    y_train = df_train['crime_level']
    X_test = df_test[features]
    y_test = df_test['crime_level']

    # Check for any remaining NaN values
    print("\nChecking for NaN values in training data:")
    print(X_train.isna().sum())
    print("\nChecking for NaN values in test data:")
    print(X_test.isna().sum())

    # Initialize and train model
    print("\nInitializing Decision Tree model...")
    model = DecisionTreeClassifier(random_state=42)
    
    # Define parameter grid
    param_grid = {
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    
    # Setup GridSearchCV
    print("Setting up GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    # Train model
    print("Starting grid search...")
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]  # Probability of class 1
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low Crime', 'Medium Crime', 'High Crime'],
                yticklabels=['Low Crime', 'Medium Crime', 'High Crime'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create classification report
    report = classification_report(y_test, y_pred, target_names=['Low Crime', 'Medium Crime', 'High Crime'])
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open(os.path.join(viz_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Plot feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(20), x='Importance', y='Feature')
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save feature importance to CSV
    feature_importance.to_csv(os.path.join(viz_dir, 'feature_importance.csv'), index=False)
    
    # Plot decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(best_model, feature_names=features, class_names=['Low Crime', 'Medium Crime', 'High Crime'],
              filled=True, rounded=True, fontsize=10)
    plt.title('Decision Tree Visualization')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'decision_tree.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save tree text representation
    tree_text = export_text(best_model, feature_names=features)
    with open(os.path.join(viz_dir, 'decision_tree.txt'), 'w') as f:
        f.write(tree_text)
    
    # Plot prediction probabilities distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=pd.DataFrame({
        'Probability': y_test_prob,
        'Actual': y_test.map({0: 'Low Crime', 1: 'Medium Crime', 2: 'High Crime'})
    }), x='Probability', hue='Actual', bins=50, kde=True)
    plt.title('Distribution of Prediction Probabilities')
    plt.xlabel('Predicted Probability of Crime Level')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'probability_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nVisualizations have been saved to: {viz_dir}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Full traceback:")
    print(traceback.format_exc()) 