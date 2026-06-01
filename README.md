# Gesture-recognition-using-EMG-signal-with-exsisting-dataset-from-kaggel.
## EMG Gesture Recognition System

This repository contains the machine learning pipeline and live inference scripts for processing Electromyography (EMG) signals to detect and classify hand/arm gestures. 

The project includes datasets for training and testing, a script to train the gesture classification model, and a real-time detection script that utilizes the trained model.

## 📂 Repository Structure

```text
EMG_project/
│
├── emg_dataset/                        # Primary training dataset
│   ├── tableConvert.com_*.csv          # Converted data tables
│   └── volunteer_*.csv                 # EMG recordings from volunteers
│
├── test_files/                         # Unseen data for model validation
│   ├── tableConvert.com_*.csv
│   └── volunteer_1.csv to 11.csv       # Test recordings across multiple subjects
│
├── emg_columns.pkl                     # Serialized feature column names
├── emg_gesture_map.pkl                 # Dictionary mapping encoded labels to gesture names
├── emg_gesture_model.pkl               # The trained machine learning model (e.g., Random Forest/SVM)
├── emg_scaler.pkl                      # Serialized scaler (e.g., StandardScaler) for data normalization
│
├── emg_training.py                     # Script to train the model and generate .pkl artifacts
└── live_emg_gesture_detection.py       # Script for real-time/live gesture inference
```

## ⚙️ Setup and Installation
Clone the repository:

Bash
```text
git clone <your-repository-url>
cd EMG_project
```
Install Required Dependencies:
Ensure you have Python installed. It is highly recommended to use a virtual environment. Install the necessary libraries (adjust based on your specific requirements):

Bash
```text
pip install pandas numpy scikit-learn
(Note: If your live detection script uses specific hardware libraries for data acquisition, add them here, e.g., pyserial).
```

## 🚀 Usage
## 1. Training the Model
To train the model from scratch using the CSV files in the emg_dataset directory:

Bash
```text
python emg_training.py
```
## What this does:
Reads and preprocesses the CSV data.
Fits a scaler to normalize the EMG signal features.
Trains the classification model.
Exports the necessary artifacts (emg_model.pkl, emg_scaler.pkl, emg_columns.pkl, and emg_gesture_map.pkl) to the root directory for later use.

## 2. Running Live Detection
Once the model and artifacts are generated, you can run the live detection script. This script loads the saved model and scaler to predict gestures on incoming data.

Bash
```text
python live_emg_gesture_detection.py
Note: Ensure your EMG hardware is properly connected and configured to stream data in the format expected by the live detection script.
```

## 🧠 Model Artifacts Explained
emg_gesture_model.pkl: The core predictive model trained on the volunteer datasets.

emg_scaler.pkl: Ensures that live incoming signals are normalized to the exact same scale as the training data, which is critical for accurate predictions.

emg_columns.pkl: Maintains the strict order and naming of features/sensors expected by the model.

emg_gesture_map.pkl: Translates the numerical outputs of the model back into human-readable gesture names (e.g., 0 -> "Fist", 1 -> "Open Hand").

## 📊 Dataset Notes
The data is separated into emg_dataset (used for training) and test_files (used for evaluation). The data consists of multi-channel EMG readings stored in CSV format, gathered from multiple volunteers to ensure the model generalizes well across different users.
