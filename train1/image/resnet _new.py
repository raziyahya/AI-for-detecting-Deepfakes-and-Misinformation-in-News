import os, cv2, keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.engine.saving import load_model
# manipulate with numpy,load with panda
import numpy as np
# import pandas as pd

# data visualization
import cv2
# import seaborn as sns



import tensorflow as tf
from tensorflow.keras import layers, Model
# from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint




# Data Import
def read_dataset(path):
    data_list = []
    label_list = []
    i=-1
    my_list = os.listdir(r'D:\germany\Dataset\New folder\Test/')
    for pa in my_list:
        i=i+1
        print(pa,"==================")
        for root, dirs, files in os.walk(r'D:\germany\Dataset\New folder\Test/' + pa):

         for f in files:
             try:
                file_path = os.path.join(r'D:\germany\Dataset\New folder\Test/'+pa, f)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                res = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
                data_list.append(res)
                # label = dirPath.split('/')[-1]
                label = i
                label_list.append(label)
             except:
                 pass
            # label_list.remove("./training")
    return (np.asarray(data_list, dtype=np.float32), np.asarray(label_list))

def read_dataset1(path):
    data_list = []
    label_list = []

    file_path = os.path.join(path)
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
    data_list.append(res)
    # label = dirPath.split('/')[-1]

            # label_list.remove("./training")
    return (np.asarray(data_list, dtype=np.float32))

# Import train_test_split
from sklearn.model_selection import train_test_split
# load dataset
x_dataset, y_dataset = read_dataset(r"D:\germany\Dataset\New folder\Test")
X_train, X_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=0)
num_classes=2
# y_train1=[]
# for i in y_train:
#     emotion = keras.utils.to_categorical(i, num_classes)
#     print(i,emotion)
#     y_train1.append(emotion)

# y_train=y_train1
x_train = np.array(X_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(X_test, 'float32')
y_test = np.array(y_test, 'float32')

x_train /= 255  # normalize inputs between [0, 1]
x_test /= 255
print("x_train.shape",x_train.shape)
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Load CIFAR-10 dataset
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# Define ResNet block
def resnet_block(x, filters, kernel_size=3, strides=1, activation='relu'):
    y = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation(activation)(y)

    y = layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(y)
    y = layers.BatchNormalization()(y)

    # Shortcut connection
    if strides > 1:
        x = layers.Conv2D(filters, kernel_size=1, strides=strides, padding='same')(x)
        x = layers.BatchNormalization()(x)

    out = layers.add([x, y])
    out = layers.Activation(activation)(out)
    return out

# Define ResNet model
def resnet(input_shape, num_classes=5):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = resnet_block(x, 32)
    x = resnet_block(x, 32)
    x = resnet_block(x, 64, strides=2)
    x = resnet_block(x, 64)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

# Create ResNet model
input_shape = x_train.shape[1:]
model = resnet(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
checkpoint_path = "training_resnet/cp.ckpt"
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True, verbose=1)



import os
import pickle
if not os.path.exists("resnet.h5"):
# Train the model
    history=model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test), callbacks=[checkpoint_callback])
    # train/validation result

    model.save("resnet.h5")
    with open('re_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)


    from matplotlib import pyplot as plt

    print(history.history.keys())
    plt.plot(history['acc'], label='accuracy')
    plt.plot(history['val_acc'], label='val_accuracy')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
else:
    import pickle
    import matplotlib.pyplot as plt

    # Read the history file
    with open('re_history.pkl', 'rb') as file:
        history = pickle.load(file)
   print(history['val_acc'])
    # Plot training and validation loss
    plt.plot(history['acc'], label='accuracy')
    plt.plot(history['val_acc'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    print(history['val_loss'])
    # Plot training and validation loss
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
# Evaluate the model
# model.evaluate(x_test, y_test)

# Make predictions

