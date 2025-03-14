import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Data Loading Functions
def load_data_from_folder(folder_path):
    """Load all CSV files from a folder containing F1 datasets"""
    dataframes = {}
    
    # List all files in the folder
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Load each CSV file
    for file in file_list:
        file_path = os.path.join(folder_path, file)
        # Extract file name without extension
        file_name = file.replace('.csv', '')
        dataframes[file_name] = pd.read_csv(file_path)
        print(f"Loaded {file_name} with {len(dataframes[file_name])} rows")
    
    return dataframes

# Feature Engineering Functions
def engineer_features(data_dict):
    """Create advanced features for F1 prediction from multiple data sources"""
    # Combine results with races, drivers and constructors
    results = data_dict['results']
    races = data_dict['races']
    drivers = data_dict['drivers']
    constructors = data_dict['constructors']
    
    # Convert position to numeric, errors='coerce' will convert non-numeric values to NaN
    results['position'] = pd.to_numeric(results['position'], errors='coerce')
    
    # Merge dataframes
    df = results.merge(races, on='raceId', how='left')
    df = df.merge(drivers, on='driverId', how='left')
    df = df.merge(constructors, on='constructorId', how='left')
    
    # If qualifying data exists, add it
    if 'qualifying' in data_dict:
        qualifying = data_dict['qualifying']
        # Convert qualifying position to numeric
        qualifying['position'] = pd.to_numeric(qualifying['position'], errors='coerce')
        df = df.merge(qualifying[['raceId', 'driverId', 'position', 'q3']], 
                      on=['raceId', 'driverId'], 
                      how='left', 
                      suffixes=('', '_qualifying'))
    
    # Add constructor standings information if available
    if 'constructor_standings' in data_dict:
        constructor_standings = data_dict['constructor_standings']
        # Convert position to numeric
        constructor_standings['position'] = pd.to_numeric(constructor_standings['position'], errors='coerce')
        df = df.merge(constructor_standings[['raceId', 'constructorId', 'points', 'position']], 
                      on=['raceId', 'constructorId'], 
                      how='left', 
                      suffixes=('', '_constructor_standings'))
    
    # Add driver standings information if available
    if 'driver_standings' in data_dict:
        driver_standings = data_dict['driver_standings']
        # Convert position to numeric
        driver_standings['position'] = pd.to_numeric(driver_standings['position'], errors='coerce')
        df = df.merge(driver_standings[['raceId', 'driverId', 'points', 'position']], 
                      on=['raceId', 'driverId'], 
                      how='left', 
                      suffixes=('', '_driver_standings'))
    
    # Convert grid positions to numeric
    df['grid'] = pd.to_numeric(df['grid'], errors='coerce')
    
    # Create driver full name
    df['driver_name'] = df['forename'] + ' ' + df['surname']
    
    # Create podium feature (1 if position <= 3, else 0)
    # Handle NaN values by filling with a large number that won't be in top 3
    df['position_filled'] = df['position'].fillna(999)
    df['podium'] = (df['position_filled'] <= 3).astype(int)
    
    # Create points finish feature (1 if position <= 10, else 0)
    df['points_finish'] = (df['position_filled'] <= 10).astype(int)
    
    # Create win feature (1 if position == 1, else 0)
    df['win'] = (df['position_filled'] == 1).astype(int)
    
    # Clean up - remove temporary column
    df.drop('position_filled', axis=1, inplace=True)
    
    # Convert date to datetime for proper sorting
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    except:
        # If date conversion fails, try a different approach
        print("Date conversion using standard format failed, trying alternative formats...")
        try:
            # Try a different date format if the first one fails
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
        except:
            # If all else fails, use the year and round to create a sortable value
            print("Using year and round as sort key instead of date")
            df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['round'].astype(str).str.zfill(2) + '-01')
    
    # Sort by date and create features based on past performance
    df = df.sort_values(['date', 'position'])
    
    # Identify the current season for each race
    df['season_year'] = df['year']
    
    # Calculate moving averages of previous results (last 3 races)
    df['last3_avg_position'] = df.groupby('driverId')['position'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    
    # Calculate podium rate in last 5 races
    df['last5_podium_rate'] = df.groupby('driverId')['podium'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    
    # Calculate win rate in last 10 races
    df['last10_win_rate'] = df.groupby('driverId')['win'].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
    
    # Calculate constructor performance (last 3 races average position)
    df['constructor_last3_avg'] = df.groupby('constructorId')['position'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    
    # Calculate qualifying vs race position difference
    if 'position_qualifying' in df.columns:
        df['qualifying_position'] = pd.to_numeric(df['position_qualifying'], errors='coerce')
        df['quali_race_diff'] = df['qualifying_position'] - df['position']
    
    # Calculate grid vs race position difference (how many positions gained/lost)
    df['grid_position_diff'] = df['grid'] - df['position']
    
    # Feature: Circuit familiarity (number of previous races at this circuit)
    df['circuit_familiarity'] = df.groupby(['driverId', 'circuitId']).cumcount()
    
    # Feature: Circuit-specific performance (avg position at this circuit)
    df['circuit_performance'] = df.groupby(['driverId', 'circuitId'])['position'].transform(
        lambda x: x.shift(1).expanding().mean())
    
    # Feature: Season performance (avg position this season)
    df['season_performance'] = df.groupby(['driverId', 'season_year'])['position'].transform(
        lambda x: x.shift(1).expanding().mean())
    
    # Extract season progress (race number within season)
    df['season_race_number'] = df.groupby('season_year').cumcount() + 1
    
    # Check for sprint race data
    if 'sprint_results' in data_dict:
        sprint_results = data_dict['sprint_results']
        # Convert sprint position to numeric
        sprint_results['position'] = pd.to_numeric(sprint_results['position'], errors='coerce')
        # Add sprint result as a feature
        df = df.merge(sprint_results[['raceId', 'driverId', 'position']], 
                      on=['raceId', 'driverId'], 
                      how='left', 
                      suffixes=('', '_sprint'))
        df['sprint_position'] = df['position_sprint']
    
    # Add pitstop information if available
    if 'pit_stops' in data_dict:
        pit_stops = data_dict['pit_stops']
        # Count number of pit stops
        pit_stop_counts = pit_stops.groupby(['raceId', 'driverId']).size().reset_index(name='pit_stop_count')
        df = df.merge(pit_stop_counts, on=['raceId', 'driverId'], how='left')
        df['pit_stop_count'] = df['pit_stop_count'].fillna(0)
    
    # Drop rows with missing values in key columns
    df_clean = df.dropna(subset=['grid', 'position'])
    
    # Fill missing values in feature columns with reasonable defaults
    df_clean['last3_avg_position'] = df_clean['last3_avg_position'].fillna(df_clean['position'].mean())
    df_clean['constructor_last3_avg'] = df_clean['constructor_last3_avg'].fillna(df_clean['position'].mean())
    
    # Create dummy variables for categorical features (limit to common circuits to avoid too many columns)
    top_circuits = df_clean['circuitId'].value_counts().nlargest(20).index
    df_clean['circuitId_top'] = df_clean['circuitId'].apply(lambda x: x if x in top_circuits else 'other')
    df_encoded = pd.get_dummies(df_clean, columns=['circuitId_top'], prefix='circuit')
    
    # Encode remaining categorical variables
    df_encoded['driver_encoded'] = df_encoded['driverId'].astype('category').cat.codes
    df_encoded['constructor_encoded'] = df_encoded['constructorId'].astype('category').cat.codes
    
    print(f"Data prepared with {len(df_encoded)} rows after feature engineering")
    return df_encoded

# Feature Selection Function
def select_features(df, target='podium'):
    """Select relevant features for the prediction model"""
    # Basic features
    features = [
        'grid',  # Starting position
        'driver_encoded',
        'constructor_encoded',
        'last3_avg_position',  # Recent form
        'last5_podium_rate',  # Podium history
        'constructor_last3_avg',  # Team performance
        'season_race_number',  # Race number in season
    ]
    
    # Add qualifying features if available
    if 'qualifying_position' in df.columns:
        features.append('qualifying_position')
    
    if 'quali_race_diff' in df.columns:
        features.append('quali_race_diff')
    
    # Add circuit familiarity and performance if available
    if 'circuit_familiarity' in df.columns:
        features.append('circuit_familiarity')
    
    if 'circuit_performance' in df.columns:
        features.append('circuit_performance')
    
    # Add season performance
    if 'season_performance' in df.columns:
        features.append('season_performance')
    
    # Add sprint race performance if available
    if 'sprint_position' in df.columns:
        features.append('sprint_position')
    
    # Add pit stop information if available
    if 'pit_stop_count' in df.columns:
        features.append('pit_stop_count')
    
    # Add win rate
    if 'last10_win_rate' in df.columns:
        features.append('last10_win_rate')
    
    # Add grid position difference
    if 'grid_position_diff' in df.columns and df['grid_position_diff'].notna().any():
        features.append('grid_position_diff')
    
    # Add circuit dummies (only include those in the dataframe)
    circuit_cols = [col for col in df.columns if col.startswith('circuit_')]
    features.extend(circuit_cols)
    
    # Ensure all selected features exist and have valid data
    valid_features = []
    for f in features:
        if f in df.columns:
            # Check if feature has valid data
            if df[f].notna().any():
                valid_features.append(f)
            else:
                print(f"Excluding feature {f} due to all values being NaN")
        else:
            print(f"Feature {f} not found in dataframe")
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ['grid', 'last3_avg_position', 'constructor_last3_avg', 
                          'last5_podium_rate', 'circuit_performance', 'season_performance']
    available_num_features = [f for f in numerical_features if f in df.columns and f in valid_features]
    
    if available_num_features:
        df[available_num_features] = scaler.fit_transform(df[available_num_features])
    
    print(f"Selected {len(valid_features)} features for model training")
    print("Features:", valid_features)
    
    # Make sure target exists in DataFrame
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in the dataframe")
    
    return df[valid_features], df[target]

# Train Model
def train_podium_model(X, y):
    """Train a random forest model to predict podium finishes"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy:.2f}')
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.2f}")
    
    # Detailed evaluation
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    # Feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    return model

# Predict Function
def predict_podiums(model, X, race_info=None):
    """Predict podium probabilities for a set of drivers"""
    # Get probabilities of podium finish
    podium_probs = model.predict_proba(X)[:, 1]
    
    # Create a DataFrame with predictions
    if race_info is not None:
        predictions = pd.DataFrame({
            'Driver': race_info['driver_name'],
            'Team': race_info['name'],
            'Grid': race_info['grid'],
            'Podium_Probability': podium_probs
        })
    else:
        predictions = pd.DataFrame({
            'Index': range(len(podium_probs)),
            'Podium_Probability': podium_probs
        })
    
    # Sort by podium probability
    predictions = predictions.sort_values('Podium_Probability', ascending=False)
    
    # Highlight likely podium finishers
    print("\nPredicted Podium Finishers:")
    print(predictions.head(3))
    
    print("\nAll Driver Predictions:")
    print(predictions)
    
    # Visualize predictions
    plt.figure(figsize=(10, 6))
    if len(predictions) > 10:
        plot_data = predictions.head(10)
    else:
        plot_data = predictions
        
    if 'Driver' in plot_data.columns:
        sns.barplot(x='Podium_Probability', y='Driver', data=plot_data)
        plt.title('Drivers by Podium Probability')
    else:
        sns.barplot(x='Podium_Probability', y='Index', data=plot_data)
        plt.title('Indices by Podium Probability')
        
    plt.tight_layout()
    plt.savefig('podium_predictions.png')
    
    return predictions

if __name__ == "__main__":
    # File path to your data folder
    data_folder = r"C:\Users\HP\Desktop\F1 Predictor\archive"
    
    try:
        # Load data
        data_dict = load_data_from_folder(data_folder)
        
        # Engineer features
        df = engineer_features(data_dict)
        
        # Check that we have the podium target
        if 'podium' not in df.columns:
            print("Error: Podium target not created. Check the data processing.")
            exit(1)
            
        # Select features and target for podium prediction
        X, y = select_features(df, target='podium')
        
        # Train model
        model = train_podium_model(X, y)
        
        # Save model
        import joblib
        joblib.dump(model, 'f1_podium_model.pkl')
        
        # Get the most recent season and race in the dataset
        max_year = df['year'].max()
        max_round = df[df['year'] == max_year]['round'].max()
        
        print(f"Most recent data is from Season {max_year}, Race {max_round}")
        
        # Get data for the most recent race
        latest_race_data = df[(df['year'] == max_year) & (df['round'] == max_round)]
        
        # Select features
        X_latest, _ = select_features(latest_race_data, target='podium')
        
        # Race info for readable output
        race_info = latest_race_data[['driver_name', 'name', 'grid']]
        
        # Predict podiums
        predictions = predict_podiums(model, X_latest, race_info)
        
        # Compare to actual results
        if 'position' in latest_race_data.columns:
            actual_results = latest_race_data[['driver_name', 'position']].sort_values('position')
            print("\nActual Results:")
            actual_podium = actual_results.head(3)
            print(actual_podium)
        
        # Save predictions to CSV
        predictions.to_csv(f'predictions_s{max_year}_r{max_round}.csv', index=False)
            
        print("Analysis complete. Visualizations saved to current directory.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()