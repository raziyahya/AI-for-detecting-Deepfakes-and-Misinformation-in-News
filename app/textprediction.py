
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences


model = tf.keras.models.load_model(r'C:\Users\Asus\Desktop\DeepFake_5_1\Deepfake\lstm_model_new (1).h5')
#
# Load the tokenizer
with open(r'C:\Users\Asus\Desktop\DeepFake_5_1\Deepfake\tokenizer_new (1).json', 'r') as f:
    tokenizer_json = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

def predict_text_function(input_text):
    print("==========================================================")
    print("==========================================================")
    print(input_text)
    print("======================================")
    print("======================================")
    # Load the trained model and tokenizer
    text_seq = tokenizer.texts_to_sequences([input_text])
    text_pad = pad_sequences(text_seq, maxlen=200, padding='post')

    # Predict using the trained model
    prediction = model.predict(text_pad)
    print(prediction)
    # Return the result

    prediction_class = 1 if prediction > 0.15 else 0  # Binary classification
    print(prediction)


    # Return response
    response = {"prediction": int(prediction_class)}
    print(response)
    import os

    # Specify the file path
    file_path = r'C:\Users\Asus\Desktop\DeepFake_5_1\Deepfake\example.txt'

    # Open the file in write mode ('w'), this will clear the file if it exists
    with open(file_path, 'w') as file:
        file.write(str(prediction[0][0])+"#"+str( int(prediction_class)))

        return response
text=""
while True:
    r=input()
    if r=="====":
        break
    else:
        text+=r
print(predict_text_function(text))
# print(predict_text_function(input()))