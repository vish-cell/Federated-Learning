#!/bin/bash

# Set paths for directories and SSH information
LSTM_MODELS_DIR="/lstm_models/"  # Directory where LSTM VM uploads models
AVERAGED_MODEL_PATH="/averaged_model/"  # Path to save the averaged model
LSTM_VM_IP="192.4.2.5"  # LSTM VM IP address
LSTM_VM_USER="root"  # Username for LSTM VM
LSTM_VM_MODEL_PATH="/updated/"  # Path to send the averaged model to LSTM VM

# Monitor the LSTM models directory for new models
inotifywait -m "$LSTM_MODELS_DIR" -e create -e moved_to |
while read path action file; do
    # Check if the file is a LSTM model (with .keras extension)
    if [[ "$file" == *.keras ]]; then
        echo "New model detected: $file"

        # Run the Python script to average the models
        echo "Averaging models..."
        python central_average.py

        # Send the averaged model back to the LSTM VM
        echo "Sending averaged model to LSTM VM..."
        scp "$/averaged_model" "$root@$192.2.4.5:$/lstm_models/"

        echo "Averaged model sent. Waiting for the next model..."
    fi
done


