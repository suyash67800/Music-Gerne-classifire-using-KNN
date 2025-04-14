# Music Genre Classification using Audio Features

This project classifies music tracks into genres using extracted audio features and the K-Nearest Neighbors (KNN) machine learning algorithm. It uses the *GTZAN* genre collection, audio processing via *Librosa, and feature-based classification with **scikit-learn*.

---

## Overview

This classification model predicts one of 10 music genres based on various audio features. Spectrograms are also generated to visualize the sound profiles.

*Genres included:*
blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

---

## Tools & Libraries Used

- Python 3.x
- Librosa (Audio Feature Extraction)
- Matplotlib (Spectrogram Visualization)
- scikit-learn (KNN, preprocessing, model evaluation)
- pandas, numpy, seaborn
- GTZAN dataset

---

## Features Extracted

From each audio file, the following features are extracted and saved:

- Chroma STFT
- RMS Energy
- Spectral Centroid
- Spectral Bandwidth
- Spectral Rolloff
- Zero Crossing Rate
- 20 MFCC coefficients

---

## Project Workflow

1. *Spectrogram Generation*  
   - Spectrogram images are created for each audio file using matplotlib.

2. *Feature Extraction*  
   - Audio features are extracted using librosa and saved into a .csv file.

3. *Data Preprocessing*  
   - Label encoding for genres
   - Feature scaling using StandardScaler
   - Train-test split (70-30)

4. *Model Training*  
   - A K-Nearest Neighbors (KNN) classifier is trained on the extracted features.

5. *Evaluation*  
   - Model is evaluated on both train and test data using:
     - Accuracy Score
     - Confusion Matrix
     - Classification Report

---

## Results

- *Training Accuracy*: ~X%  
- *Testing Accuracy*: ~Y%  
(You can replace X and Y with your actual results)
