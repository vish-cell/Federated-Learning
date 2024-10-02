#!/bin/bash

# Loop until all datasets are processed
for i in {1..12}; do
    echo "Training LSTM model on dataset data_$i.csv"
    
    # Run the Python script for LSTM training
    python lstm_training.py
    
    # After the model is trained and saved, wait for the averaged model to be received
    echo "Waiting for averaged model from central VM..."
    while [ ! -f "path_to_received_averaged_model/averaged_model.keras" ]; do
        sleep 5  # Wait for 5 seconds before checking again
    done
    
    echo "Averaged model received. Proceeding to next dataset..."
done
