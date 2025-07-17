import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

#import time

#test1
def train_and_evaluate_lstm(input_csv, model_save_path, 
                            scaler_save_path, 
                            num_epochs=60, 
                            batch_size=16, 
                            test_size=0.2, 
                            random_state=42, 
                            progress_callback=None, 
                            date_from=None, 
                            date_to=None,
                            use_solar=False):
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    def find_fully_nan_columns(input_csv):
        data = pd.read_csv(input_csv)
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        columns_to_drop = []
        
        # Find the datetime column case-insensitively
        datetime_col = next((col for col in data.columns if col.lower() == "date time"), None)

        if not datetime_col:
            raise ValueError("No 'datetime' column found (case-insensitive match failed).")

        # Normalize column to lowercase name "datetime" for downstream use
        data['date time'] = pd.to_datetime(data[datetime_col], errors='coerce')
        
        # Drop original column if it's not already lowercase 'datetime'
        if datetime_col != 'date time':
            data.drop(columns=[datetime_col], inplace=True)
        
        if date_from and date_to:
            data = data[(data['date time'] >= pd.Timestamp(date_from)) & 
                    (data['date time'] <= pd.Timestamp(date_to))]
            print(f"Filtered data between {date_from} and {date_to}. Remaining rows: {len(data)}")
        
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            if data[col].isnull().all():
                columns_to_drop.append(col)

        return columns_to_drop, data

    def hunt_nans(input_csv):
        # Drop fully NaN columns
        columns_to_drop, data = find_fully_nan_columns(input_csv)
        remaining_data = data.drop(columns=columns_to_drop)
        

        # Define a threshold for the maximum allowed percentage of NaNs
        threshold = 0.5
        #Step 1: Drop columns with too many NaNs
        min_non_nany = int((1 - threshold) * remaining_data.shape[0])
        masky = remaining_data.notna().sum(axis=0) >= min_non_nany
        remaining_data = remaining_data.loc[:, masky]

        # Step 2: Drop rows with too many NaNs (based on now-pruned columns)
        min_non_nanx = int((1 - threshold) * remaining_data.shape[1])
        maskx = remaining_data.notna().sum(axis=1) >= min_non_nanx
        remaining_data = remaining_data.loc[maskx]
        

        if use_solar:
            # Find target column(s) that contain "Solar Radiation"
            solar_radiation_cols = [col for col in remaining_data.columns if "solar radiation" in col.lower()]
            if not solar_radiation_cols:
                raise ValueError("No column found containing 'Solar Radiation'.")
            target_col = solar_radiation_cols[0]
        else:
            # Find target column(s) that contain "Wind Speed"
            wind_speed_cols = [col for col in remaining_data.columns if "wind speed" in col.lower()]
            if not wind_speed_cols:
                raise ValueError("No column found containing 'Wind Speed'.")
            target_col = wind_speed_cols[0]
        
        y = remaining_data[target_col].values
        X = remaining_data.drop(columns=[target_col])
        X = X.drop(columns=['date time'], errors='ignore')


        print(f"X Shape: {X.shape}")
        print(f"y Shape: {y.shape}")

        return X, y, remaining_data
    
    X, y, remaining_data = hunt_nans(input_csv)
    print("Finished loading and processing data")

    #Split the data into training and testing sets
    # Use 80% of the data for training and 20% for testing
    X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False)
    print("X_train_data shape:", X_train_data.shape)
    print("X_test_data shape:", X_test_data.shape)

    # Fill NaN values with the mean of each column    
    for col in X_train_data.columns:
        if X_train_data[col].isna().sum() > 0:
            X_train_data[col] = X_train_data[col].fillna(X_train_data[col].mean())

    for col in X_test_data.columns:
        if X_test_data[col].isna().sum() > 0:
            X_test_data[col] = X_test_data[col].fillna(X_test_data[col].mean())

    #Convert y_train_data and y_test_data to 2D arrays
    y_train_data = pd.Series(y_train_data.flatten())
    y_train_data = y_train_data.fillna(y_train_data.mean()).values.reshape(-1, 1)
    
    y_test_data = pd.Series(y_test_data.flatten())
    y_test_data = y_test_data.fillna(y_test_data.mean()).values.reshape(-1, 1)
    

    #Scale the data
    xscaler = StandardScaler()
    yscaler = StandardScaler()
    X_train_scaled = xscaler.fit_transform(X_train_data)
    X_test_scaled = xscaler.transform(X_test_data)
    
    y_train_data = yscaler.fit_transform(y_train_data)
    y_test_data = yscaler.transform(y_test_data)
    #Create sequences
    def create_sequences(X, y, seq_length):
        Xs, ys = [], []
        for i in range(len(X) - seq_length):
            Xs.append(X[i:i+seq_length])
            ys.append(y[i+seq_length])
        return np.array(Xs), np.array(ys)

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_data, seq_length=24)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_data, seq_length=24)
    X_train_seq = np.array(X_train_seq, dtype=np.float64)

    print("NaNs in X_train:", np.isnan(X_train_seq).sum())
    print("NaNs in y_train:", np.isnan(y_train_seq).sum())

    # Save the scaler for future use
    joblib.dump(xscaler, scaler_save_path)
    joblib.dump(yscaler, scaler_save_path.replace('.pkl', '_y.pkl'))
    print("Finished saving scalers")

    # Convert data to PyTorch tensors and move to the device (GPU or CPU)
    X_train = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train_seq, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test_seq, dtype=torch.float32).to(device)

    # Create DataLoader for batch processing
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Finished creating DataLoaders")

    # Define the LSTM model
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.2):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
            self.fc = nn.Linear(hidden_size, 1)  # Output 1 value (Wind Speed)
        
        def forward(self, x):
            h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
            c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
            out, _ = self.lstm(x, (h_0, c_0))
            out = self.fc(out[:, -1, :])  # Take output from the last time step
            return out

    input_size = X_train.shape[2]  # Number of features
    hidden_size = 16  # Number of LSTM units
    num_layers = 1  # Number of LSTM layers

    model = LSTMModel(input_size, hidden_size, num_layers).to(device)

    # Define loss function and optimizer with L2 regularization
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=5e-4)

    # Train the model
    train_losses = []
    test_losses = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_test_loss += loss.item()
        
        avg_test_loss = running_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        scheduler.step(avg_test_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, LR = {optimizer.param_groups[0]["lr"]:.6f}')

        if progress_callback:
            progress_callback(train_losses[:], test_losses[:])

    # Save the model
    torch.save(model.state_dict(), model_save_path)

    # Save losses to a CSV file
    loss_df = pd.DataFrame({'epoch': range(1, num_epochs+1), 'train_loss': train_losses, 'test_loss': test_losses})
    loss_df.to_csv('training_losses.csv', index=False)

    # Evaluate the model using additional metrics

    # Make predictions on the test set
    model.eval()
    y_test_pred = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            y_test_pred.extend(outputs.cpu().numpy())

    y_test_pred = np.array(y_test_pred)
    y_test_np = y_test.cpu().numpy()

    # Make predictions on the training set
    y_train_pred = []
    with torch.no_grad():
        for inputs, _ in train_loader:
            outputs = model(inputs)
            y_train_pred.extend(outputs.cpu().numpy())

    y_train_pred = np.array(y_train_pred)
    y_train_np = y_train.cpu().numpy()

    y_test = yscaler.inverse_transform(y_test.cpu().numpy())
    y_test_pred = yscaler.inverse_transform(y_test_pred)
    
    
    corr, _ = pearsonr(y_test.flatten(), y_test_pred.flatten())
    print("Pearson Correlation:", corr)

    # Calculate R^2, MAE, and RMSE for the test set
    y_test_np_unscaled = yscaler.inverse_transform(y_test_np)
    r2_test = r2_score(y_test_np_unscaled, y_test_pred)
    mae_test = mean_absolute_error(y_test_np_unscaled, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test_np_unscaled, y_test_pred))


    # Calculate R^2, MAE, and RMSE for the training set
    y_train_np_unscaled = yscaler.inverse_transform(y_train_np)
    r2_train = r2_score(y_train_np_unscaled, y_train_pred)
    mae_train = mean_absolute_error(y_train_np_unscaled, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train_np_unscaled, y_train_pred))

    # Calculate baseline R^2 using Linear Regression
    r2_baseline = LinearRegression().fit(X_train_scaled, y_train_data).score(X_test_scaled, y_test_data)
    print(f"Baseline Linear Regression R²: {r2_baseline:.4f}")
    r2_persist = r2_score(y_test_np_unscaled[1:], y_test_np_unscaled[:-1])
    print(f'Persistence R²: {r2_persist:.4f}')


    # Print the results
    print(f'R^2 Score (Test): {r2_test:.4f}')
    print(f'Mean Absolute Error (MAE) (Test): {mae_test:.4f}')
    print(f'Root Mean Squared Error (RMSE) (Test): {rmse_test:.4f}')
    print(f'R^2 Score (Train): {r2_train:.4f}')
    print(f'Mean Absolute Error (MAE) (Train): {mae_train:.4f}')
    print(f'Root Mean Squared Error (RMSE) (Train): {rmse_train:.4f}')

    # Save the evaluation metrics to a CSV file
    metrics_df = pd.DataFrame({
        'metric': ['R^2 Score (Test)', 'Mean Absolute Error (Test)', 'Root Mean Squared Error (Test)', 
                   'R^2 Score (Train)', 'Mean Absolute Error (Train)', 'Root Mean Squared Error (Train)'],
        'value': [r2_test, mae_test, rmse_test, r2_train, mae_train, rmse_train]
    })
    metrics_df.to_csv('evaluation_metrics.csv', index=False)

    return train_losses, test_losses, remaining_data, input_size






def predict(model_path,
            scaler_path,
            input_csv,
            output_csv,
            date_from,
            date_to,
            feature_size,
            use_solar=False,
            plot_path=None):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
 
    # Define the LSTM model class
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.3):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
            self.fc = nn.Linear(hidden_size, 1)  # Adjust the output size if you're predicting more than one target
        
        def forward(self, x):
            h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
            c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
            out, _ = self.lstm(x, (h_0, c_0))
            out = self.fc(out[:, -1, :])  # Take output from the last time step
            return out

    # Load the trained model
    input_size = feature_size  # Adjust according to your number of features
    hidden_size = 16  # Same as in training
    num_layers = 1  # Same as in training
    model = LSTMModel(input_size, hidden_size, num_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load the scaler
    xscaler = joblib.load(scaler_path)
    yscaler = joblib.load(scaler_path.replace('.pkl', '_y.pkl'))
    data = input_csv
    # Find the datetime column case-insensitively
    datetime_col = next((col for col in data.columns if col.lower() == "date time"), None)

    if not datetime_col:
        raise ValueError("No 'datetime' column found (case-insensitive match failed).")

    # Normalize column to lowercase name "datetime" for downstream use
    data['date time'] = pd.to_datetime(data[datetime_col])
    
    # Drop original column if it's not already lowercase 'datetime'
    if datetime_col != 'date time':
        data.drop(columns=[datetime_col], inplace=True)


    # Load new data for prediction
    new_data = input_csv
    print('Finished loading new data')

    # --- Calculate date range in terms of duration and relative offset ---
    duration = date_to - date_from
    print(f"Original duration: {duration}")

    # --- Find the last year in the dataset ---
    latest_year = data['date time'].dt.year.max()
    print(f"Latest year in dataset: {latest_year}")

    # Align the target prediction window with the same month/day but latest year
    target_from = date_from.replace(year=latest_year)
    target_to = target_from + duration

    target_from = pd.to_datetime(target_from)
    target_to = pd.to_datetime(target_to)

    # --- Filter the prediction input ---
    pred_df = data[(data['date time'] >= target_from) & (data['date time'] <= target_to)].copy()
    print(f"Prediction window: {target_from.date()} to {target_to.date()}")
    print(f"Filtered rows: {len(pred_df)}")

    def create_inference_sequences(X, seq_length):
        sequences = []
        for i in range(len(X) - seq_length):
            sequences.append(X[i:i+seq_length])
        return np.array(sequences)

    # Prepare the input data for prediction
    X_new = pred_df[xscaler.feature_names_in_].copy()

    X_new = X_new.ffill().bfill()  # Fill NaNs with forward and backward fill

    # Standardize the features
    X_new_scaled = xscaler.transform(X_new)

    X_new_scaled_seq = create_inference_sequences(X_new_scaled, seq_length=24)

    X_new_tensor = torch.tensor(X_new_scaled_seq, dtype=torch.float32).to(device)
    batch_size = 16
    dataset = TensorDataset(X_new_tensor)
    loader = DataLoader(dataset, batch_size=batch_size)
    # Make predictions
    predictions = []
    with torch.no_grad():
        for batch in loader:
            X_batch = batch[0].to(device)
            y_batch_pred = model(X_batch)
            predictions.append(y_batch_pred.cpu())
    
    # Concatenate all batches
    predictions = torch.cat(predictions, dim=0).numpy()   
    valid_dates = pred_df['date time'].values[24:]  # Skip first 24 to match the number of sequences
    
    #Reshape predictions to match the expected output
    predictions = yscaler.inverse_transform(predictions)
    
    if use_solar:
        prediction_df = pd.DataFrame({
            'date time': valid_dates,
            'Predicted Solar Radiation': predictions.flatten()
        })
    else:
        prediction_df = pd.DataFrame({
            'date time': valid_dates,
            'Predicted Wind Speed': predictions.flatten()
        })    
        
    prediction_df = prediction_df.sort_values('date time')
    prediction_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

    return prediction_df
