#!/bin/bash

# Variables
LSTM_VM_IP="192.4.2.5"  # LSTM VM's IP address
LSTM_USER="lstm_user"    # LSTM VM's username
CENTRAL_MODEL_DIR="/path/to/central/lstm_folder"  # Directory to store received LSTM models
AVERAGED_MODEL_NAME="averaged_model.h5"  # Filename for the averaged model
MODEL_COUNT=1  # Number of LSTM models expected per round (since only 1 LSTM model is expected per round)

# Function to average models
average_models() {
    echo "Averaging models..."
    
    # List all models received in the CENTRAL_MODEL_DIR
    MODELS=($CENTRAL_MODEL_DIR/*.h5)

    # Ensure there are models to average
    if [ ${#MODELS[@]} -eq 0 ]; then
        echo "No models found to average."
        return
    fi

    # Create a Python script to handle averaging
    cat << EOF > average_models.py
import numpy as np
import os
from tensorflow.keras.models import load_model

def average_models(models_list, output_model_name):
    # Load all models and extract weights
    all_weights = [load_model(model).get_weights() for model in models_list]
    
    # Initialize the averaged weights
    avg_weights = [np.zeros(w.shape) for w in all_weights[0]]
    
    # Sum the weights of all models
    for weights in all_weights:
        avg_weights = [avg_w + w for avg_w, w in zip(avg_weights, weights)]
    
    # Divide by the number of models to compute the average
    avg_weights = [avg_w / len(models_list) for avg_w in avg_weights]
    
    # Load the first model's architecture to save the averaged weights
    model = load_model(models_list[0])
    model.set_weights(avg_weights)
    
    # Save the averaged model
    model.save(output_model_name)
    print(f"Averaged model saved as {output_model_name}")

if __name__ == "__main__":
    model_directory = "${CENTRAL_MODEL_DIR}"
    model_files = [f for f in os.listdir(model_directory) if f.endswith('.h5')]
    model_paths = [os.path.join(model_directory, f) for f in model_files]
    average_models(model_paths, "${CENTRAL_MODEL_DIR}/${AVERAGED_MODEL_NAME}")
EOF

    # Run the Python script to average models
    python3 average_models.py
    
    # Clean up the temporary averaging script
    rm average_models.py

    echo "Model averaging completed. Averaged model saved as $AVERAGED_MODEL_NAME"
}

# Loop to continuously check for new models from the LSTM VM
while true; do
    # Step 1: Check if the required number of models have been received
    NUM_MODELS=$(ls -1q $CENTRAL_MODEL_DIR/*.h5 | wc -l)
    
    if [ "$NUM_MODELS" -ge "$MODEL_COUNT" ]; then
        echo "All models for the round received. Averaging models..."
        
        # Step 2: Perform model averaging
        average_models
        
        # Step 3: Send the averaged model back to the LSTM VM
        echo "Sending averaged model back to LSTM VM..."
        scp $CENTRAL_MODEL_DIR/$AVERAGED_MODEL_NAME $LSTM_USER@$LSTM_VM_IP:/path/to/lstm/models/
        
        # Step 4: Clean up for the next round
        echo "Cleaning up model files from Central VM..."
        rm $CENTRAL_MODEL_DIR/*.h5  # Remove the individual models
        
        echo "Round completed. Waiting for the next round of models..."
    else
        echo "Waiting for models from LSTM VM. Currently received: $NUM_MODELS/$MODEL_COUNT"
    fi
    
    # Sleep for 30 seconds before checking again
    sleep 30
done
