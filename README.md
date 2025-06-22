# Digit Classifier App

This project is a Streamlit-based web application for digit recognition using machine learning models trained on the MNIST dataset. Users can draw digits, upload images, or use their webcam to predict handwritten digits.

## Features

- **Draw Digits:**
  - Use an interactive canvas to draw digits and make predictions.

- **Webcam Digit Recognition:**
  - Use your webcam to capture handwritten digits in real-time.
  - The app detects digits in the video stream, predicts them, and overlays the result on the video.

- **Image Upload & Batch Prediction:**
  - Upload one or more images for batch digit prediction.
  - Optionally upload or input true labels to evaluate accuracy and view a confusion matrix.

## How to Run

1. **Prerequisites**
- Python 3.10+
- Install dependencies with pip or uv:
  ```bash
  pip install -r requirements.txt
  ```
  ```bash
  uv sync
  ```

2. **Start the App**
   ```bash
   streamlit run app.py
   ```

## Models
Pretrained models are stored in the `models` directory. The app loads the selected model for predictions.

## License
MIT License
