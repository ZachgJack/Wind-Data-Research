from LSTM_Prototype import train_and_evaluate_lstm
from K_means_Plotter import run_kmeans_plotter

LaunchLSTM = False
KMeansPlotter = False

Choice = input("Do you want to launch LSTM  or K-means plotter? ")

if Choice.lower() == "lstm":
    LaunchLSTM = True
elif Choice.lower() == "k-means":
    KMeansPlotter = True

if LaunchLSTM: 
    
    csv_path = input(r"Paste the path of the data:")  # or read_csv or whatever format
    model_save_path = (r"C:\Users\Rover\OneDrive\Documents\lstm_model.pt")  # or read_csv or whatever format
    scalar_save_path = (r"C:\Users\Rover\OneDrive\Documents\scaler.pkl")  # or read_csv or whatever format
   
    train_and_evaluate_lstm(input_csv=csv_path, 
        model_save_path=model_save_path,
        scaler_save_path=scalar_save_path,
        num_epochs=60,
        batch_size=32,
        test_size=0.2,
        random_state=42)
    
if KMeansPlotter: 
    csv_path = input(r"Paste the path of the data:")  # or read_csv or whatever format
    run_kmeans_plotter(csv_path)
