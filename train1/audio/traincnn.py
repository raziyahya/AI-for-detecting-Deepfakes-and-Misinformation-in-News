import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the dataset
# Assuming your CSV file is named 'audio_features.csv' and has the features and labels.
data = pd.read_csv('DATASET-balanced.csv')

# Assuming 'label' is the column containing the target ('FAKE' or 'REAL')
X = data.drop(columns=['LABEL']).values  # Features
y = data['LABEL'].values  # Labels ('FAKE' or 'REAL')

# Step 2: Convert string labels ('FAKE', 'REAL') to numerical values (0, 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # 'REAL' -> 0, 'FAKE' -> 1

# Step 3: Normalize the features (important for CNNs)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Reshape the input data for CNN (reshape to 4D)
# Reshape to (samples, height, width, channels) for CNN input
# X_scaled.shape is (num_samples, num_features), and we need to reshape it to (num_samples, num_features, 1, 1)
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1, 1)  # 4D shape for CNN input
print(X_scaled.shape)
# Step 5: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Step 6: Convert labels to categorical (one-hot encoding) for binary classification
y_train_cat = to_categorical(y_train, num_classes=2)
y_test_cat = to_categorical(y_test, num_classes=2)


# Step 7: Define the CNN model
def create_cnn_model(input_shape):
    model = Sequential()
    # First convolutional layer
    model.add(Conv2D(32, (1, 1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(1,1)))

    # Second convolutional layer
    model.add(Conv2D(64, (1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,1)))

    # Third convolutional layer
    model.add(Conv2D(128, (1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,1)))

    # Flatten the output for dense layers
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout for regularization
    model.add(Dense(64, activation='relu'))

    # Output layer for binary classification (fake or real)
    model.add(Dense(2, activation='softmax'))  # Use softmax for multi-class classification

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Step 8: Create the model
input_shape = (X_train.shape[1], 1, 1)  # (features, 1 height, 1 width)
model = create_cnn_model(input_shape)

# Step 9: Train the model
history = model.fit(X_train, y_train_cat, epochs=20, batch_size=32, validation_data=(X_test, y_test_cat))

# Step 10: Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print("==========================")
print("==========================")
print(history.history)

import matplotlib.pyplot as plt

# Plotting training and validation accuracy
plt.figure(figsize=(12, 6))  # Adjust figure size if needed
plt.subplot(1, 2, 1)  # Create a subplot (1 row, 2 columns, 1st plot)
plt.plot(history.history['acc'], label='Train Accuracy')
plt.plot(history.history['val_acc'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Audio Accuracy CNN')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)  # 2nd plot in the same row
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Audio Loss CNN')
plt.legend()

# Adding main heading for the figure
plt.suptitle('Model Accuracy and Loss during Training', fontsize=16)

# Display the plots
plt.tight_layout()
plt.show()
