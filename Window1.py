import sys
import traceback
import queue
import pandas as pd
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QLineEdit, QPlainTextEdit
from PySide6.QtCore import Qt, Signal, QObject, QThread, QTimer
from PySide6.QtGui import QPalette, QColor
from ui_Main_Window import Ui_MainWindow 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from LSTM_Prototype import train_and_evaluate_lstm, predict

# This is a worker class that will run the LSTM training in a separate thread
# to keep the GUI responsive. It will also handle output redirection to a queue
# so that the terminal widget can display the output in real-time.
class LSTMWorker(QObject):
    finished_signal = Signal()
    plot_data_signal = Signal(list, list)
    epoch_update_signal = Signal(list, list)

    def __init__(self, file_path, model_path, scaler_path, output_queue, date_from=None, date_to=None):
        super().__init__()
        self.file_path = file_path
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.output_queue = output_queue
        self.date_from = date_from
        self.date_to = date_to

    def run(self):
        sys.stdout = QueueStream(self.output_queue)
        sys.stderr = QueueStream(self.output_queue)

        def emit_epoch(train_losses, test_losses):
            self.epoch_update_signal.emit(train_losses, test_losses)

        try:
            train_losses, test_losses, remaining_data, input_size = train_and_evaluate_lstm(
                self.file_path,
                model_save_path=self.model_path,
                scaler_save_path=self.scaler_path,
                num_epochs=60,
                batch_size=32,
                test_size=0.2,
                random_state=42,
                progress_callback=emit_epoch,
                date_from=self.date_from,
                date_to=self.date_to,
                
            )
            print("Training completed.")
            # AFTER LSTM training
            predictions = predict(
                model_path=self.model_path,
                scaler_path=self.scaler_path,
                input_csv=remaining_data,
                output_csv="prediction_output.csv",
                date_from=self.date_from,
                date_to=self.date_to,
                feature_size=input_size
            )
            # Load the processed CSV file
            df = remaining_data

            # Slice to the same date range as the predictions
            df["date time"] = pd.to_datetime(df["date time"])  # or whatever column
            mask = (df["date time"] >= pd.to_datetime(self.date_from)) & (df["date time"] <= pd.to_datetime(self.date_to))
            y_true = df.loc[mask, "Wind Speed"].tolist()  # Replace with actual column name

            # Plotting: directly access the main window and call its method
            MainWindow.instance.update_prediction_plot(y_true, predictions['Predicted Wind Speed'].tolist())

        except Exception as e:
            traceback.print_exc()
        
        self.plot_data_signal.emit(train_losses, test_losses)
        self.finished_signal.emit()

# This is the main window class that sets up the GUI and handles user interactions.
# It includes a terminal widget for user input and output, a drop area for CSV files,
# and a loss_canvas for plotting the training results.
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        MainWindow.instance = self
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.centralwidget.setAcceptDrops(True)
        self.output_queue = queue.Queue()
        self.queue_timer = QTimer()
        self.queue_timer.setInterval(50)  # 50 ms polling
        self.queue_timer.timeout.connect(self.flush_terminal_queue)
        self.queue_timer.start()

        # Initialize the target type
        self.terminal = TerminalWidget()
        self.terminal.user_input_signal.connect(self.handle_terminal_input)
        self.ui.terminalPlaceholder.layout().addWidget(self.terminal)
        self.terminal.setStyleSheet("background-color: black; color: lime; font-family: monospace;")

        # Create a loss_canvas for plotting
        self.loss_canvas = MplCanvas(self)
        self.pred_canvas = MplCanvas(self)
        self.ui.lossPlaceholder.layout().addWidget(self.loss_canvas)
        self.ui.predictPlaceholder.layout().addWidget(self.pred_canvas)
       
        # Create a drop area for CSV files
        self.drop_area = DropArea(terminal_output_widget=self.terminal, parent=self.ui.centralwidget)
        self.ui.dropPlaceholder.layout().addWidget(self.drop_area)
        

        self.ui.pushButton.clicked.connect(self.run_action)

    # This method is called when the user clicks the button to run the LSTM training.
    # It checks if a file has been dropped, initializes the LSTM worker,
    # and starts the training in a separate thread.
    def run_action(self):
        if not self.drop_area.file_path:
            self.terminal.append_output("No file has been dropped yet.")
            return

        model_save_path = r"C:\Users\Rover\OneDrive\Documents\GUITest\lstm_model.pt"
        scalar_save_path = r"C:\Users\Rover\OneDrive\Documents\GUITest\scaler.pkl"
        from_date = self.ui.dateFrom.date().toPython()
        to_date = self.ui.dateTo.date().toPython()

        # Create a new thread and worker for LSTM training
        self.thread = QThread()
        self.worker = LSTMWorker(
            self.drop_area.file_path,
            model_save_path,
            scalar_save_path,
            self.output_queue,
            date_from=from_date,
            date_to=to_date
        )
        
        # Connect signals to slots
        self.worker.moveToThread(self.thread)
        self.worker.plot_data_signal.connect(self.update_loss_plot)
        self.worker.epoch_update_signal.connect(self.update_loss_plot)
        self.worker.finished_signal.connect(self.thread.quit)
        self.worker.finished_signal.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

        self.terminal.append_output("⏳ Running LSTM training in background...")   

    def handle_terminal_input(self, user_input):
        user_input = user_input.strip().lower()

        if user_input == "wind":
            self.target_type = "wind_speed"
            self.terminal.append_output("Using wind speed as the LSTM target.")
        elif user_input == "solar":
            self.target_type = "solar_radiation"
            self.terminal.append_output("Using solar radiation as the LSTM target.")
        else:
            self.terminal.append_output("Invalid input. Please enter 'wind' or 'solar'.")

    def update_loss_plot(self, train_losses, test_losses):
        self.loss_canvas.ax.clear()
        self.loss_canvas.ax.plot(train_losses, label='Train Loss')
        self.loss_canvas.ax.plot(test_losses, label='Test Loss')
        self.loss_canvas.ax.set_title('Model Loss')
        self.loss_canvas.ax.set_xlabel('Epoch')
        self.loss_canvas.ax.set_ylabel('Loss')
        self.loss_canvas.ax.legend()
        self.loss_canvas.draw()

    def update_prediction_plot(self, y_true, y_pred):
        self.pred_canvas.ax.clear()
        self.pred_canvas.ax.plot(y_true, label="True")
        self.pred_canvas.ax.plot(y_pred, label="Predicted")
        self.pred_canvas.ax.set_title("Predictions vs Actual")
        self.pred_canvas.ax.set_xlabel("Time Step")
        self.pred_canvas.ax.set_ylabel("Target Value")
        self.pred_canvas.ax.legend()
        self.pred_canvas.draw()


    def flush_terminal_queue(self):
        while not self.output_queue.empty():
            text = self.output_queue.get()
            self.terminal.append_output(text)
   
# This is the terminal widget that allows user input and displays output.
# It includes a text area for output and a line edit for user input.
class TerminalWidget(QWidget):
    user_input_signal = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)

        self.input = QLineEdit()
        self.input.returnPressed.connect(self._handle_user_input)

        layout = QVBoxLayout()
        layout.addWidget(self.output)
        layout.addWidget(self.input)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def _handle_user_input(self):
        text = self.input.text().strip()
        self.append_output(f"> {text}")
        self.input.clear()
        self.user_input_signal.emit(text)

    def append_output(self, text):
        self.output.appendPlainText(text)
    
    
# This is the drop area widget that allows users to drag and drop CSV files.
# It validates the file type and displays the file name and dimensions if valid.
# If an invalid file is dropped, it shows an error message.
class DropArea(QLabel):
    def __init__(self, terminal_output_widget=None, parent=None):
        super().__init__(parent)
        self.terminal = terminal_output_widget
        self.setText("Drop a .csv file here")
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setAutoFillBackground(True)
        self._set_default_style()

        self.file_path = None  

    def _set_default_style(self):
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#f0f0f0"))
        self.setPalette(palette)
        self.setStyleSheet("border: 2px dashed #aaa; font-size: 16px;")

    def _set_hover_style(self):
        self.setStyleSheet("border: 2px solid green; background-color: #eaffea; font-size: 16px;")

    def _set_error_style(self):
        self.setStyleSheet("border: 2px solid red; background-color: #ffeaea; font-size: 16px;")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if not url.toLocalFile().endswith('.csv'):
                    self._set_error_style()
                    return 
                
                if url.toLocalFile().endswith('.csv'):
                    event.acceptProposedAction()
                    self._set_hover_style()
                    return
                
        event.ignore()
    
    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._set_default_style()

    def dropEvent(self, event):
        self._set_default_style()
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                print(f"DROPPED: {file_path}")  # Debug
                if file_path.endswith('.csv'):
                    self.file_path = file_path  # Save path
                    self.handle_csv(file_path)
                    return
        self.setText("Not a CSV file.")
        self._set_error_style()

    # Handles the CSV file, reads it, and updates the terminal output.
    # If the file contains solar energy data, it prompts the user to select a target.
    def handle_csv(self, path):
        try:
            df = pd.read_csv(path)
            self.df = df  # Save DataFrame for later use
            self.setText(f"Loaded: {path.split('/')[-1]}\n{df.shape[0]} rows × {df.shape[1]} columns")
            for col in df.columns:
                if "solar energy" in col.lower():
                    if self.terminal:
                        self.terminal.append_output("CSV loaded. Detected targets: wind_speed, solar_radiation.")
                        self.terminal.append_output("Which target would you like to use for the LSTM? (wind/solar)")
                    break  
                
        except Exception as e:
            print(f"Failed to read file:\n{e}")
            self._set_error_style()

# This is the loss_canvas class that integrates with Matplotlib to display plots.
# It creates a Figure and an Axes, and provides a method to draw the plot.
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure()
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
      
# This is a stream class that redirects output to a queue.
# It allows the LSTM worker to send output to the terminal widget in real-time.
class QueueStream:
    def __init__(self, output_queue):
        self.queue = output_queue

    def write(self, text):
        if text.strip():
            self.queue.put(text)

    def flush(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
