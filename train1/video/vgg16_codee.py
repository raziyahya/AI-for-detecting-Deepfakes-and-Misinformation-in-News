# # Import libraries
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.layers import Dense, Dropout, Flatten
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# # Define data paths
# train_data_dir = r'C:\Users\mani\Desktop\dataset\archive\Messidor-2+EyePac_Balanced'
# validation_data_dir = r'C:\Users\mani\Desktop\dataset\archive\Messidor-2+EyePac_Balanced'
#
# # Define image dimensions
# img_width, img_height = 224, 224
#
# # Data augmentation for training data
# train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
#
# # Load training and validation data
# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=32,
#     class_mode='categorical'
# )
#
# validation_generator = ImageDataGenerator(rescale=1./255)
# validation_generator = validation_generator.flow_from_directory(
#     validation_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=32,
#     class_mode='categorical'
# )
#
# # Load pre-trained VGG16 model with ImageNet weights
# # Include top layers for transfer learning
# base_model = VGG16(weights='imagenet', include_top=True, input_shape=(img_width, img_height, 3))
#
# # Freeze a specific number of pre-trained layers
# num_to_freeze = 5  # Adjust this based on your dataset and experiment
#
# for layer in base_model.layers[:num_to_freeze]:
#   layer.trainable = False
#
# # Add a new classifier head
# x = base_model.output
# x = Flatten(name='my_flatten_layer')(x)  # Unique name for Flatten layer
# x = Dense(1024, activation='relu')(x)
# x = Dropout(0.5)(x)
# predictions = Dense(len(train_generator.classes), activation='softmax')(x)
#
# # Create the final model
# model = Model(inputs=base_model.input, outputs=predictions)
#
# # Compile the model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # Train the model
# model.fit(
#     train_generator,
#     epochs=10,  # Adjust number of epochs
#     validation_data=validation_generator
# )
#
# # Save the trained model
# model.save('vgg16_fine_tuned_model.h5')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# loading the directories
training_dir = r'D:\germany\VideoDataset\new\Train/'
validation_dir = r'D:\germany\VideoDataset\new\Val/'
test_dir = r'D:\germany\VideoDataset\new\Test/'


image_files = glob(training_dir + '/*/*.jp*g')
valid_image_files = glob(validation_dir + '/*/*.jp*g')


folders = glob(training_dir + '/*')
num_classes = len(folders)
print ('Total Classes = ' + str(num_classes))

from keras.models import Model
from keras.layers import Flatten, Dense
from keras.applications import VGG16
#from keras.preprocessing import image

IMAGE_SIZE = [64, 64]  # we will keep the image size as (64,64). You can increase the size for better results.

# loading the weights of VGG16 without the top layer. These weights are trained on Imagenet dataset.
vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)  # input_shape = (64,64,3) as required by VGG

# this will exclude the initial layers from training phase as there are already been trained.
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
#x = Dense(128, activation = 'relu')(x)   # we can add a new fully connected layer but it will increase the execution time.
x = Dense(num_classes, activation = 'softmax')(x)  # adding the output layer with softmax function as this is a multi label classification problem.

model = Model(inputs = vgg.input, outputs = x)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator




from keras.applications.vgg16 import preprocess_input

training_datagen = ImageDataGenerator(
                                    rescale=1./255,   # all pixel values will be between 0 an 1
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    preprocessing_function=preprocess_input)

validation_datagen = ImageDataGenerator(rescale = 1./255, preprocessing_function=preprocess_input)

training_generator = training_datagen.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = 200, class_mode = 'categorical')
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size = IMAGE_SIZE, batch_size = 200, class_mode = 'categorical')


print(training_generator.class_indices)


training_images = 5400
validation_images = 1200

history = model.fit(training_generator,
                   steps_per_epoch = 20,  # this should be equal to total number of images in training set. But to speed up the execution, I am only using 10000 images. Change this for better results.
                   epochs = 20,  # change this for better results
                   validation_data = validation_generator,
                   validation_steps = 10)

from matplotlib import pyplot as plt
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Video Model VGG16')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

from matplotlib import pyplot as plt
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Video Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()