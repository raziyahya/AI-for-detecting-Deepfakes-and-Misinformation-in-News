import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the dataset
data = pd.read_csv('DATASET-balanced.csv')  # Replace with your dataset path

# Assuming 'label' is the column containing the target ('FAKE' or 'REAL')
X = data.drop(columns=['LABEL']).values  # Features
y = data['LABEL'].values  # Labels ('FAKE' or 'REAL')

# Step 2: Convert string labels ('FAKE', 'REAL') to numerical values (0, 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # 'REAL' -> 0, 'FAKE' -> 1

# Step 3: Normalize the features (important for CNNs)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Reshape the input data for CNN + LSTM (needs to be 3D)
# Reshape to (samples, time_steps, features) for CNN + LSTM input
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)  # 3D shape for CNN + LSTM input

# Step 5: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Step 6: Convert labels to categorical (one-hot encoding) for binary classification
y_train_cat = to_categorical(y_train, num_classes=2)
y_test_cat = to_categorical(y_test, num_classes=2)


# Step 7: Define the CNN + LSTM model
def create_cnn_lstm_model(input_shape):
    model = Sequential()

    # CNN layers
    model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))  # Convolutional layer
    model.add(MaxPooling1D(2))  # Max pooling layer

    model.add(Conv1D(128, 3, activation='relu'))  # Convolutional layer
    model.add(MaxPooling1D(2))  # Max pooling layer

    # LSTM layer
    model.add(LSTM(64, return_sequences=False))  # LSTM layer

    # Dense layers
    model.add(Dropout(0.5))  # Dropout for regularization
    model.add(Dense(64, activation='relu'))  # Fully connected layer
    model.add(Dense(2, activation='softmax'))  # Output layer for binary classification

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Step 8: Create the model
input_shape = (X_train.shape[1], 1)  # (time_steps, features) for CNN + LSTM input
model = create_cnn_lstm_model(input_shape)

# Step 9: Train the model
history = model.fit(X_train, y_train_cat, epochs=20, batch_size=32, validation_data=(X_test, y_test_cat))

# Step 10: Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print(history.history)
# Step 11: Visualize the training history (accuracy)
# plt.plot(history.history['acc'], label='Train Accuracy')
# plt.plot(history.history['val_acc'], label='Test Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.show()
import matplotlib.pyplot as plt

# Plotting training and validation accuracy
plt.figure(figsize=(12, 6))  # Adjust figure size if needed
plt.subplot(1, 2, 1)  # Create a subplot (1 row, 2 columns, 1st plot)
plt.plot(history.history['acc'], label='Train Accuracy')
plt.plot(history.history['val_acc'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Audio Accuracy CNN_LSTM')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)  # 2nd plot in the same row
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Audio Loss CNN_LSTM')
plt.legend()

# Adding main heading for the figure
plt.suptitle('Model Accuracy and Loss during Training', fontsize=16)

# Display the plots
plt.tight_layout()
plt.show()
