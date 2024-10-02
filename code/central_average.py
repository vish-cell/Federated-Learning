  GNU nano 5.6.1                                                         central_average.py                                                                   
import os
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Directory containing LSTM models from the LSTM VM
lstm_models_directory = '/lstm_models/'  # Path to the folder where the models from LSTM VM are saved
averaged_model_path = 'averaged_model/averaged_model.keras'  # Path to save the averaged model

def average_models(model_list):
    """ Average the weights of the models in the list """
    averaged_weights = [np.mean([model.get_weights()[i] for model in model_list], axis=0) for i in range(len(model_list[0].get_weights()))]
    
    # Create a new model using the same architecture as the first model
    model_config = model_list[0].get_config()  # Get the architecture configuration
    averaged_model = Sequential.from_config(model_config)  # Recreate the model from the config
    averaged_model.set_weights(averaged_weights)  # Set the averaged weights
    
    return averaged_model

# Step 1: Load all LSTM models from the directory
model_list = []
for filename in os.listdir(lstm_models_directory):
    if filename.endswith(".keras"):
        model_path = os.path.join(lstm_models_directory, filename)
        model_list.append(load_model(model_path))

# Step 2: Average the models
if model_list:
    averaged_model = average_models(model_list)
    os.makedirs(os.path.dirname(averaged_model_path), exist_ok=True)
    averaged_model.save(averaged_model_path)
    print(f"Averaged model saved at {averaged_model_path}")
else:
    print("No models found for averaging.")

