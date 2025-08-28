# Korrecto

Korrecto is a multimodal AI-driven system designed to help users perform fitness exercises. The system combines real-time pose detection, visual and vocal feedback, and machine learning, to offer an intelligent, interactive coaching experience that improves training quality and reduces the risk of injury.

## Features

- **Pose detection** using [MediaPipe](https://developers.google.com/mediapipe) to track human skeleton landmarks from webcam input.   
- **Real-time feedback** on the correctness of performed exercises.  
- Supports the following exercises:  
  - Squats  
  - Push Ups  
  - Jumping Jacks  

## Project Structure
- **/analysis:** dataset creation and ML model training.
- **/feedback:** feedback logic.
- **/skeleton:** pose detection and exercise checks logic.
- **/ui:** graphical interface setup.

## Libraries used
- mediapipe 0.10.21
- numpy 1.26.4 (must be <2 for mediapipe compatibility)
- python 3.10
- opencv 4.5.5.64
- pandas 2.3.1
- scikit-learn 1.7.1
- tensorflow 2.19.0
- SpeechRecognition 3.14.3
- pyaudio 0.2.14
- pyttsx3 2.99
- PyQt6 6.9.1

## Authors
Developed by Adriana Pia Barbalace and Alessia Castronovo.
