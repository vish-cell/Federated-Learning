#!/bin/bash

# Set paths for directories and SSH information
DATA_DIRECTORY="data/"  # Directory containing CSV files for training
LSTM_MODELS_DIRECTORY="lstm_models/"  # Directory to save trained models
CENTRAL_SERVER_IP="192.4.2.5"  # Central server IP address
CENTRAL_SERVER_USER="root"  # Username for the central server
CENTRAL_MODEL_PATH="/path_to_central_models/"  # Path on the central server to save the model

# Ensure the model directory exists
mkdir -p "$LSTM_MODELS_DIRECTORY"

# Loop through the first 11 CSV files for training
for i in $(seq 1 11); do
    CSV_FILE="$(ls $DATA_DIRECTORY | grep '.csv' | sed -n "${i}p")"
    
    if [ -n "$CSV_FILE" ]; then
        echo "Training with dataset: $CSV_FILE"
        
        # Run the Python script to train the model and send it to the central server
        python train_and_send.py "$DATA_DIRECTORY$CSV_FILE" "$LSTM_MODELS_DIRECTORY" "$CENTRAL_SERVER_IP" "$CENTRAL_SERVER_USER" "$CENTRAL_MODEL_PATH"

        echo "Model training completed for: $CSV_FILE"
    else
        echo "No more CSV files to process."
        break
    fi
done
