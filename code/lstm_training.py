import sys
import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
import paramiko

def train_model(csv_file, model_save_path):
    # Load the dataset
    data = pd.read_csv(csv_file)
    
    # Preprocess the data as needed (for demonstration, assuming the data is already prepared)
    # ...

    # Define the model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, data.shape[1])))  # Adjust input shape
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Adjust output layer according to your task

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Fit the model
    # Assume X_train and y_train are defined after preprocessing
    # model.fit(X_train, y_train, epochs=100, verbose=0)

    # Save the model
    model_save_file = os.path.join(model_save_path, f'model_{os.path.basename(csv_file).split(".")[0]}.keras')
    model.save(model_save_file)

    return model_save_file

def send_model_to_central(model_file, central_ip, central_user, central_path):
    # Use paramiko to send the model to the central server
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(central_ip, username=central_user)

    sftp = ssh.open_sftp()
    sftp.put(model_file, os.path.join(central_path, os.path.basename(model_file)))
    sftp.close()
    ssh.close()

if __name__ == "__main__":
    csv_file = sys.argv[1]
    model_save_path = sys.argv[2]
    central_ip = sys.argv[3]
    central_user = sys.argv[4]
    central_model_path = sys.argv[5]

    model_file = train_model(csv_file, model_save_path)
    send_model_to_central(model_file, central_ip, central_user, central_model_path)
