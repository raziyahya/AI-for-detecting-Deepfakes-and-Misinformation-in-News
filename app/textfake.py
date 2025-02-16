import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline


# Load the dataset
# For simplicity, assume ai_generated.txt contains AI-generated text, and human_written.txt contains human-written text

def load_data():
    ai_texts = []
    human_texts = []

    # Load AI-generated texts
    with open("ai_generated.txt", "r") as f:
        ai_texts = f.readlines()

    # Load human-written texts
    with open("human_written.txt", "r") as f:
        human_texts = f.readlines()

    # Create a DataFrame
    data = pd.DataFrame({
        "text": ai_texts + human_texts,
        "label": [1] * len(ai_texts) + [0] * len(human_texts)  # 1 for AI-generated, 0 for human-written
    })

    return data


# Preprocess data and vectorize it using TF-IDF
def preprocess_and_vectorize(data):
    X = data['text']
    y = data['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# Build and train the model
def train_model(X_train, y_train):
    # Use a pipeline with TF-IDF and Logistic Regression
    model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    model.fit(X_train, y_train)

    return model


# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


# Main function
def main():
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_and_vectorize(data)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
