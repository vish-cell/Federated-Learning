import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Function to load data from the CSV file
def load_data(dataset_path):
    data = pd.read_csv(dataset_path)
    data['Label'] = data['Label'].astype('category').cat.codes
    X = data.drop(columns=['Label'])
    y = data['Label']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    return X_reshaped, y

# Function to create the LSTM model
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model
def train_model(model, X, y, output_path):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    model.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM model")
    parser.add_argument('--dataset', type=str, required=True, help="Path to dataset CSV")
    parser.add_argument('--weights', type=str, help="Path to pre-trained weights (optional)")
    parser.add_argument('--output', type=str, required=True, help="Path to save the trained model")
    
    args = parser.parse_args()
    
    # Load data
    X, y = load_data(args.dataset)
    num_classes = len(np.unique(y))
    
    # Load existing model or create a new one
    if args.weights:
        model = load_model(args.weights)
    else:
        model = create_model((X.shape[1], X.shape[2]), num_classes)
    
    # Train model
    train_model(model, X, y, args.output)
