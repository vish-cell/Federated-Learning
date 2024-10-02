#!/bin/bash

# Variables
CENTRAL_VM_IP="192.4.2.4"  # Central VM's IP address
CENTRAL_USER="central_user"  # Central VM's username
LSTM_MODEL_DIR="/path/to/lstm/models"  # Directory to store LSTM models
CENTRAL_MODEL_DIR="/path/to/central/lstm_folder"  # Central directory for averaging models
DATA_DIR="/path/to/lstm/data"  # Directory with CSV files
MODEL_NAME="lstm_model_round.h5"  # Name for the trained model file
AVERAGED_MODEL_NAME="averaged_model.h5"  # Name for the averaged model file
TRAINING_SCRIPT="/path/to/train_lstm.py"  # Python script for training

# Loop through each CSV file in the data directory
for CSV_FILE in $DATA_DIR/*.csv; do
    echo "Training on dataset $CSV_FILE..."

    # Step 1: Train LSTM model on the current dataset
    python3 $TRAINING_SCRIPT --dataset $CSV_FILE --output $LSTM_MODEL_DIR/$MODEL_NAME

    # Step 2: Send the trained model to Central VM
    echo "Sending trained LSTM model to Central VM..."
    scp $LSTM_MODEL_DIR/$MODEL_NAME $CENTRAL_USER@$CENTRAL_VM_IP:$CENTRAL_MODEL_DIR

    # Step 3: Wait for the averaged model from Central VM
    echo "Waiting for the averaged model from Central VM..."
    while ! scp $CENTRAL_USER@$CENTRAL_VM_IP:$CENTRAL_MODEL_DIR/$AVERAGED_MODEL_NAME $LSTM_MODEL_DIR/$AVERAGED_MODEL_NAME; do
        echo "Averaged model not yet available. Retrying in 30 seconds..."
        sleep 30
    done

    # Step 4: Continue training with the averaged model on the next round
    echo "Continuing training with averaged model..."
    python3 $TRAINING_SCRIPT --dataset $CSV_FILE --weights $LSTM_MODEL_DIR/$AVERAGED_MODEL_NAME --output $LSTM_MODEL_DIR/$MODEL_NAME

    echo "Round completed for dataset $CSV_FILE"
done

echo "Training completed on all datasets!"
