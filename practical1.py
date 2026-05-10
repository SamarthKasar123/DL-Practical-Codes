import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler

# Load dataset
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

# Normalize data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build model
model = Sequential()

model.add(Dense(64, input_dim=13, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile model
model.compile(
    loss='mean_squared_error',
    optimizer='adam'
)

# Train model
model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    verbose=0
)

# Evaluate model
mse = model.evaluate(X_test, y_test, verbose=0)

print("Mean Squared Error:", mse)
