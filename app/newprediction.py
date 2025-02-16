
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json


def predict_image(image_path):
    with open(r'D:\germany\Deep Fake\Deepfake\alexnew.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    # Load the model weights
    loaded_model.load_weights(r"D:\germany\Deep Fake\Deepfake\alexnew.weights.h5")

    img_width, img_height=100,100
    image = load_img(image_path, target_size=(img_width, img_height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize the image (if not done already)

    # Make prediction
    class_labels = ['Fake', 'Real']  # Class labels from the generator

    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[0][predicted_class_index]

    return class_labels[predicted_class_index], confidence

# Example usage
test_image_path = r"D:\germany\Deep Fake\Deepfake\media\fake_10000.jpg"  # Replace with your test image path

predicted_class, confidence = predict_image(test_image_path)
print(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f})")
