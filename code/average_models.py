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
    # Define the directory containing the LSTM models
    model_directory = "/path/to/central/lstm_folder"  # Update this path
    averaged_model_name = "averaged_model.h5"  # Filename for the averaged model
    
    # List all models in the specified directory
    model_files = [f for f in os.listdir(model_directory) if f.endswith('.h5')]
    
    # Create full paths for the model files
    model_paths = [os.path.join(model_directory, f) for f in model_files]

    # Ensure there are models to average
    if len(model_paths) > 0:
        average_models(model_paths, os.path.join(model_directory, averaged_model_name))
    else:
        print("No models found to average.")
