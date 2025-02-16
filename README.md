# AI for Detecting Misinformation and Deepfakes in News

## Project Overview
This project successfully developed AI models for detecting deepfakes in *image, **audio, **text, and **video* formats, specifically targeting media within the news domain to combat misinformation. The system includes a user-friendly interface for file uploads and detection result visualization.

---

## Datasets
•⁠  ⁠*Video*: [DDeep Fake Detection (DFD) Entire Original Dataset](https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset)
•⁠  ⁠*Audio*: [DEEP-VOICE: DeepFake Voice Recognition](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition)
•⁠  ⁠*Image*: [Deepfake and Real Images Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)
•  *Text* : [TweepFake - Twitter deep Fake text Dataset](https://www.kaggle.com/datasets/mtesconi/twitter-deep-fake-text)


---

## Methodology
•⁠  ⁠*Tools and Frameworks*: TensorFlow, Keras.

### *Image Detection*
•⁠  ⁠*Models Tried*: AlexNet, DenseNet-121, ResNet18
•⁠  ⁠*Selected Model*: ResNet18 (achieved the best accuracy of 97.84%).


### *Audio Detection*
•⁠  ⁠*Models Tried*: CNN VGG, LSTM, CNN 
•⁠  ⁠*Selected Model*: CNN VGG (for its highest accuracy of 97.84%).


### *Video Detection*
•⁠  ⁠*Models Tried*:
  - EfficientNetB0, VGG16
•⁠  ⁠*Selected Model*: EfficientNetB0 (achieved the best accuracy of 99.46%).


### *Text Detection*
•⁠  ⁠Initially attempted AI-generated text detection but found it inaccurate.
•⁠  ⁠Successfully implemented *LSTM+NLP-based web scraping* for text analysis which gave better results for ai generated news.

---

## User Interface
•⁠  ⁠Developed using *Django*.
•  Added Signup and Login page.
•⁠  ⁠Fully integrated with image, audio, text, and video detection models.
•⁠  For reference, results are stored in the ⁠ 'UI-Results' ⁠ and accuracy & loss graph in 'Training_Results' folder.

---

## Conclusion
This project demonstrated the feasibility of AI models in detecting deepfakes across multiple media formats, providing a robust tool for combating misinformation in news media. 

---
