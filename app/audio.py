import librosa
import numpy as np


# Function to extract features from an audio file
def extract_audio_features(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

    # RMS (Root Mean Square Error)
    rms = librosa.feature.rms(y=y)

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    # Spectral Rolloff (corrected argument name)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)

    # Zero-Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)

    # MFCCs (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # Combine all features into a dictionary or return as numpy array
    features = {
        'chroma_stft': np.mean(np.mean(chroma_stft, axis=1)),
        'rms': np.mean(np.mean(rms, axis=1)),
        'spectral_centroid': np.mean(np.mean(spectral_centroid, axis=1)),
        'spectral_bandwidth': np.mean(np.mean(spectral_bandwidth, axis=1)),
        'spectral_rolloff': np.mean(np.mean(spectral_rolloff, axis=1)),
        'zero_crossing_rate': np.mean(np.mean(zero_crossing_rate, axis=1)),
        'mfcc': np.mean(mfccs, axis=1)  # MFCCs 1 to 20
    }
    print(features['chroma_stft'])
    # Flatten the dictionary values to return a single vector of features
    feature_vector = np.hstack([features['chroma_stft'],
                                features['rms'],
                                features['spectral_centroid'],
                                features['spectral_bandwidth'],
                                features['spectral_rolloff'],
                                features['zero_crossing_rate'],
                                features['mfcc']])

    return np.asarray([feature_vector])


def predict_audio_code(audio_path):
    features = extract_audio_features(audio_path)
    print(type(features))
    features=features.reshape(features.shape[0], features.shape[1], 1, 1)
    # Print the extracted features
    print("Extracted Audio Features:")
    print(features)
    from tensorflow.keras.models import load_model
    from tensorflow.keras.initializers import glorot_uniform

    loaded_model = load_model('vgg_audio_model.h5')
    # loaded_model = load_model('vgg_audio_model.h5', custom_objects={'GlorotUniform': glorot_uniform})

    # Load the saved model
    # loaded_model = load_model('vgg_audio_model.h5')
    res=loaded_model.predict(features)
    predicted_class_index = np.argmax(res[0])

    print(res,predicted_class_index)
    return str(predicted_class_index),res[0][predicted_class_index]*100
