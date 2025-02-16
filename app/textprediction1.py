
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = tf.keras.models.load_model(r'D:\germany\Deep Fake\Deepfake\lstm_model_new (1).h5')
#
# Load the tokenizer
with open(r'D:\germany\Deep Fake\Deepfake\tokenizer_new (1).json', 'r') as f:
    tokenizer_json = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)


# Define the prediction view
def predict_text(text):
    print(len(text))
    # Get the input text from the request (e.g., from a POST request)


    # Tokenize the input text
    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_seq, maxlen=200, padding='post')

    # Predict using the trained model
    prediction = model.predict(text_pad)
    print(prediction)
    # Return the result
    result = "AI Generated" if prediction > 0.5 else "Normal Text"
    print(result)



print(predict_text(''' 
Twelve of the 56 officials detained this year held roles in the central Communist Party and state agencies—double the number in 2023. This indicates a growing emphasis on targeting corruption within the top echelons of the party apparatus and ministries.
Since launching his anti-corruption drive in 2012, Xi has pursued a relentless campaign to root out both high-level “tigers” and low-level “flies.” The military, particularly the PLA Rocket Force responsible for overseeing China’s nuclear arsenal, has been a focal point of this effort.
'''))