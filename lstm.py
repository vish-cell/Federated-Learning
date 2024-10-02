import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Load the data
data = pd.read_csv('dataset\inputs\data_4.csv')

# Step 2: Preprocess the data
# Assuming 'Label' is the target and the rest are features
# Convert labels to numeric values if they are categorical
data['Label'] = data['Label'].astype('category').cat.codes

# Features and target variable
X = data.drop(columns=['Label'])
y = data['Label']

# Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Reshape the data for LSTM
# LSTM expects input in the shape of (samples, time steps, features)
# Here, we can treat each sample as having 1 time step
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Step 4: Create the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_reshaped.shape[1], X_reshaped.shape[2])))

# Optionally add dropout for regularization
model.add(Dropout(0.2))

# Output layer
model.add(Dense(len(data['Label'].unique()), activation='softmax'))  # Use 'sigmoid' for binary classification

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_reshaped, y, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Step 6: Evaluate the model
loss, accuracy = model.evaluate(X_reshaped, y)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Optionally save the model
model.save('model.keras')

