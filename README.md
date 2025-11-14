# 🎭 Facial Emotion Recognition (FER) Using Deep Learning

This project implements a Facial Emotion Recognition model that predicts human emotions from facial images using a Convolutional Neural Network (CNN).
The model is trained on the FER-2013 dataset and can classify the following 7 emotions:

1.)Angry.

2.)Disgust.

3.)Fear

4.)Happy

5.)Sad

6.)Surprise

7.)Neutral

## 📂 Dataset
This project uses the FER-2013 dataset:
- 35,887 images
- 48×48 grayscale
- 7 emotion categories
  
The dataset is widely used for benchmarking facial emotion models and is available on Kaggle.

## 🧠 Model Architecture
The CNN model includes:
- Multiple Conv2D + ReLU layers
- Batch Normalization
- MaxPooling layers
- Dropout for regularization
- Dense layers
- Final Softmax for 7 emotion classes

## 🚀 Features
- Trained CNN for emotion classification
- Supports image uploads for prediction
- Clean preprocessing pipeline
- Ready-to-run on Google Colab
- Includes sample inference code

## 🎯 Applications
This model can be used for:
- Real-time emotion-based human-computer interaction
- Sentiment-aware UI/UX systems
- Classroom/office mood monitoring
- Mental health analytics
- Customer behavior analysis

## 🔮 Future Scope

1) Live Emotion Detection (Real-Time Webcam System)
Use OpenCV / Mediapipe to detect faces and classify emotions live:
Capture webcam video
Detect face
Predict emotion in every frame
Display results in real time

2) Emotion-Based Music Recommendation
Integrate with:
Spotify API
YouTube Music API
Examples:
Emotion	Recommended Music
Happy	Energetic + upbeat songs
Sad	Soft, calming tracks
Angry	Relaxing instrumentals
Fear	Motivational tracks
Neutral	Lofi / chill beats

3) Emotion-Based Movie Recommendation
Use TMDB / IMDb datasets to suggest movies based on mood:
Happy → Comedies, feel-good films
Sad → Inspirational / comforting movies
Fear → Horror, thrillers
Neutral → Dramas, classics

4) Full Emotion-Aware AI Assistant
Combine:
Emotion recognition
Voice interaction
Personalized recommendations
To create a smart emotional AI companion.

## 📊 Model Performance Graphs (Accuracy & Loss)

The following graphs are generated during training:

✔ Training vs Validation Accuracy

Shows how well the model learns emotion patterns over epochs.

✔ Training vs Validation Loss

Displays convergence and overfitting detection.
<img width="981" height="451" alt="image" src="https://github.com/user-attachments/assets/fd0d9643-4fe3-4d19-8a2e-01666a9893f8" />


