

from keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json

def AlexNet(input_shape, num_classes):
  """Defines the AlexNet architecture.

  Args:
      input_shape: A tuple of 3 integers representing the input image shape (height, width, channels).
      num_classes: An integer representing the number of output classes.

  Returns:
      A Keras Model object representing the AlexNet architecture.
  """
  # Define input layer
  inputs = layers.Input(shape=input_shape)

  # Convolutional layers
  conv1 = layers.Conv2D(filters=64, kernel_size=11, strides=4, padding="same", activation="relu")(inputs)
  pool1 = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(conv1)
  norm1 = layers.BatchNormalization()(pool1)  # Batch normalization (not in original AlexNet)

  conv2 = layers.Conv2D(filters=192, kernel_size=5, padding="same", activation="relu")(norm1)
  pool2 = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(conv2)
  norm2 = layers.BatchNormalization()(pool2)

  conv3 = layers.Conv2D(filters=384, kernel_size=3, padding="same", activation="relu")(norm2)
  conv4 = layers.Conv2D(filters=384, kernel_size=3, padding="same", activation="relu")(conv3)
  conv5 = layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu")(conv4)
  pool3 = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(conv5)
  norm3 = layers.BatchNormalization()(pool3)

  # Fully-connected layers
  flatten = layers.Flatten()(norm3)
  fc1 = layers.Dense(4096, activation="relu")(flatten)
  dropout1 = layers.Dropout(0.5)(fc1)  # Dropout layer (not in original AlexNet)
  fc2 = layers.Dense(4096, activation="relu")(dropout1)
  dropout2 = layers.Dropout(0.5)(fc2)  # Dropout layer (not in original AlexNet)
  predictions = layers.Dense(num_classes, activation="softmax")(dropout2)

  # Define model
  model = Model(inputs=inputs, outputs=predictions)
  return model

# Training Example (replace placeholders with your data and hyperparameters)
train_data_dir = r"C:\Users\Asus\Desktop\AI_Project\Dataset\Deep fake and real images\Dataset\Train"
val_data_dir = r"C:\Users\Asus\Desktop\AI_Project\Dataset\Deep fake and real images\Dataset\Test"
img_width, img_height = 100,100
num_classes = 2  # Adjust based on your dataset
epochs = 30
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data generators (optional)
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode="categorical"
)
val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode="categorical"
)

import numpy as np

num_samples = len(val_generator)
batch_size = val_generator.batch_size

# Initialize empty lists to store batches of data
x_train = []
y_train = []

# Loop through the generator and extract batches of data
for i in range(num_samples):
    batch_x, batch_y = val_generator.__next__()


    for j in range(0,len(batch_x)):
        r=batch_x[j]
        x_train.append(batch_x[j])
        y_train.append(batch_y[j])

x_train=np.asarray(x_train)
# Create AlexNet model
model = AlexNet((img_width, img_height, 3), num_classes)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
import os
import pickle
import numpy as np
if not os.path.exists("alexnew.weights.h5"):
# Train the model
    history=model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )



# Assuming 'history' is your history object
    with open('history.pkl', 'wb') as file:
        pickle.dump(history.history, file)


# Save the model architecture as JSON
    model_json = model.to_json()
    with open("alexnew.json", "w") as json_file:
        json_file.write(model_json)

    # Save the model weights
    model.save_weights("alexnew.weights.h5")


    # model.save("resnet.h5")
    with open('re_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)
    # model.save("alexnetmodel.h5")

    from matplotlib import pyplot as plt
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
else:
    import pickle
    import matplotlib.pyplot as plt

    # Load the model architecture from JSON
    with open('alexnew.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    # Load the model weights
    loaded_model.load_weights("alexnew.weights.h5")

    res = loaded_model.predict(x_train)
    po = []
    for i in range(0, len(x_train)):
        print(res[i], y_train[i])
        print(np.argmax(res[i]))
        po.append(np.argmax(res[i]))
    y_train1=[]
    for i in y_train:
        y_train1.append(np.argmax(i))

    from sklearn.metrics import classification_report

    res = classification_report(y_train1, po)
    print(res)

    from sklearn.metrics import confusion_matrix

    res = confusion_matrix(y_train1, po)
    print(res)

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Example confusion matrix data
    conf_matrix = res

    # Define labels for the matrix
    labels = ["Class 1", "Class 2"]

    # Plotting the confusion matrix
    sns.set(font_scale=1.2)  # Adjust font size
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.show()

    # Read the history file
    with open('history.pkl', 'rb') as file:
        history = pickle.load(file)

    # Plot training and validation loss
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    # Plot training and validation loss
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
# Prediction Example (replace with your test image path)
# test_image_path = "path/to/test/image.jpg"
# test_image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(img_width, img_height))
# test_image = tf.keras.preprocessing.image
