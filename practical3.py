import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# Load dataset
dataset = pd.read_csv('Google_Stock_Price_Train.csv')

# Select Open column
training_set = dataset.iloc[:, 1:2].values

# Normalize data
scaler = MinMaxScaler(feature_range=(0,1))
training_set_scaled = scaler.fit_transform(training_set)

# Create training data
X_train = []
y_train = []

for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape data for RNN
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build RNN model
model = Sequential()

model.add(SimpleRNN(units=50, activation='tanh', input_shape=(X_train.shape[1], 1)))

model.add(Dense(1))

# Compile model
model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# Train model
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32
)

print("RNN Model Trained Successfully")
