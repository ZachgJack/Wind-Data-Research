import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def run_kmeans_plotter(input_csv):
    
    def find_fully_nan_columns(input_csv):
            data = pd.read_csv(input_csv)
            data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
            columns_to_drop = []

            data['Date time'] = pd.to_datetime(data['Date time'], errors='coerce')
            data['hour'] = data['Date time'].dt.hour
            data['day_of_year'] = data['Date time'].dt.dayofyear

            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                if data[col].isnull().all():
                    columns_to_drop.append(col)

            return columns_to_drop, data

    def hunt_nans(input_csv):
        # Drop fully NaN columns
        columns_to_drop, data = find_fully_nan_columns(input_csv)
        remaining_data = data.drop(columns=columns_to_drop)

        # Eliminate stragling NaNs
        # Define a threshold for the maximum allowed percentage of NaNs
        threshold = 0.2
        # Step 1: Drop columns with too many NaNs
        min_non_nany = int((1 - threshold) * remaining_data.shape[0])
        masky = remaining_data.notna().sum(axis=0) >= min_non_nany
        remaining_data = remaining_data.loc[:, masky]

        # Step 2: Drop rows with too many NaNs (based on now-pruned columns)
        min_non_nanx = int((1 - threshold) * remaining_data.shape[1])
        maskx = remaining_data.notna().sum(axis=1) >= min_non_nanx
        remaining_data = remaining_data.loc[maskx]

        # Find target column(s) that contain "Wind Speed"
        wind_speed_cols = [col for col in remaining_data.columns if "wind speed" in col.lower()]

        # Check if any were found
        if not wind_speed_cols:
            raise ValueError("No column found containing 'Wind Speed'.")

        # Use the first matching column as target
        target_col = wind_speed_cols[0]
        y = remaining_data[target_col].values
        X = remaining_data.drop(columns=[target_col])

        print(f"X Shape: {X.shape}")
        print(f"y Shape: {y.shape}")

        return X, y

    X, y = hunt_nans(input_csv)

    # Fill remaining NaN values with the mean of each column
    for col in X.columns:
        if X[col].isna().sum() > 0:
            X[col] = X[col].fillna(X[col].mean())

    y = pd.Series(y.flatten())
    y = y.fillna(y.mean()).values.reshape(-1, 1)
    # Remove columns with less than 5 unique values
    X = X.loc[:, X.nunique() >= 5]


    #Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    y_scaled_df = pd.DataFrame(y_scaled, columns=['Wind Speed'])
    Xy_scaled = pd.concat([X_scaled_df, y_scaled_df], axis=1)

    print(Xy_scaled.columns.tolist())
    comparison = input("Choose a feature to compare with Wind Speed (e.g., 'hour', 'day_of_year'): ")

    if comparison in Xy_scaled.columns:
        try:
            selected_data = Xy_scaled[[comparison, 'Wind Speed']]
            kmeans = KMeans(n_clusters=3)
            kmeans.fit(selected_data)
            Xy_scaled['kmeans_3'] = kmeans.labels_
        except Exception as e:
            print(f"Error during KMeans fitting: {e}")
    else:
        print(f"Column '{comparison}' not found in the dataset.")

    plt.figure(figsize=(8, 6))
    plt.scatter(Xy_scaled[comparison], Xy_scaled['Wind Speed'], c=Xy_scaled['kmeans_3'], cmap='viridis', alpha=0.6)
    plt.xlabel(f'{comparison} (scaled)')
    plt.ylabel('Wind Speed (scaled)')
    plt.title('KMeans Clustering (k=3)')
    plt.grid(True)
    plt.show()

