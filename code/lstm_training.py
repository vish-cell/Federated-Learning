import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Set the path to the data directory and model directory
data_directory = 'data'  # Directory containing the CSV files
model_directory = 'models'  # Directory to save the models
averaged_model_path = 'path_to_received_averaged_model/averaged_model.keras'  # Path to the averaged model received from central VM

# Ensure model directory exists
os.makedirs(model_directory, exist_ok=True)

# Step 2: Train the model on each dataset in the data directory
for i in range(1, 13):  # Assuming you have 12 CSV files
    data_path = os.path.join(data_directory, f'data_{i}.csv')  # Adjust as per your filenames
    data = pd.read_csv(data_path)

    # Preprocess the data
    data['Label'] = data['Label'].astype('category').cat.codes
    X = data.drop(columns=['Label'])
    y = data['Label']
    
    # Normalize the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape the data for LSTM
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_reshaped.shape[1], X_reshaped.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(len(data['Label'].unique()), activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_reshaped, y, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    
    # Save the model with a unique name
    model_filename = os.path.join(model_directory, f'lstm_{i}.keras')  # Naming convention
    model.save(model_filename)
    
    # Optional: wait for the averaged model to be received from the central VM
    while not os.path.exists(averaged_model_path):
        print("Waiting for the averaged model from the central VM...")
        
# At this point, you can load the averaged model if needed
