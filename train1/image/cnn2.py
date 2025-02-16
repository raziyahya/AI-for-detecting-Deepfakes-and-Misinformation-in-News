from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.models import model_from_json

# Define image dimensions and number of classes (adjust as needed)
img_width, img_height = 28, 28
num_classes = 2

# Data paths for training and validation sets (replace with your paths)
train_data_dir = r"C:\Users\Asus\Desktop\AI_Project\Dataset\Deep fake and real images\Dataset\Train"
val_data_dir = r"C:\Users\Asus\Desktop\AI_Project\Dataset\Deep fake and real images\Dataset\Test"

# Data augmentation (optional)
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

# Create training and validation generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,  # Adjust batch size as needed
    class_mode="categorical"
)
val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,  # Adjust batch size as needed
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

# Create the model (replace architecture as needed)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
import os
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# history=model.fit(train_generator, epochs=30, validation_data=val_generator)
# print(history.history.keys())
if not os.path.exists("dsharp_model_weights.h5" ):
# Train the model (adjust epochs and hyperparameters as needed)
    history=model.fit(train_generator, epochs=35, validation_data=val_generator)
    #
    # # Make predictions on new images (replace with your image path)
    # new_image_path = r"C:\Users\ASUS\OneDrive\Desktop\dataset new"
    # # Preprocess the image (adjust preprocessing steps as needed)
    # img = ...  # Load and preprocess the image
    # prediction = model.predict(np.expand_dims(img, axis=0))[0]
    # predicted_class = np.argmax(prediction)

    # print(f"Predicted class: {predicted_class}")

    import os
    import pickle

    # Assuming 'history' is your history object
    with open('d_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)

    # Save the model architecture as JSON
    model_json = model.to_json()
    with open("dsharp_model.json", "w") as json_file:
        json_file.write(model_json)

    # Save the model weights
    # model.save_weights("dsharp_model_weights.h5") code changed 15.11.2024
    model.save_weights("dsharp_model_weights.weights.h5")


    # model.save("dsharp.h5")

    from matplotlib import pyplot as plt
    print(history.history.keys())
    plt.plot(history.history['accuracy']) #changed15.11.2024
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
else:
    import pickle
    import matplotlib.pyplot as plt

    # Load the model architecture from JSON
    with open('dsharp_model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    # Load the model weights
    # loaded_model.load_weights("dsharp_model_weights.h5") changed
    model.save_weights("dsharp_model_weights.weights.h5")
    res = loaded_model.predict(x_train)
    po = []
    for i in range(0, len(x_train)):
        print(res[i], y_train[i])
        print(np.argmax(res[i]))
        po.append(np.argmax(res[i]))
    from sklearn.metrics import classification_report

    y_train1 = []
    for i in range(0, len(x_train)):
        print(res[i], y_train[i])

        y_train1.append(np.argmax(y_train[i]))
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
    with open('d_history.pkl', 'rb') as file:
        history = pickle.load(file)
    print(history)
    # Plot training and validation loss
    plt.plot(history['accuracy'], label='accuracy')         #changed
    plt.plot(history['val_accuracy'], label='val_accuracy') #changed
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