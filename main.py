import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('mnist_model_3.keras')


# Function to preprocess the image for prediction
def preprocess(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding to create a binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get the largest contour (assumed to be the digit)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Extract the digit region
        digit_region = gray[y:y + h, x:x + w]
        # Resize the digit to a fixed size (28x28)
        resized_digit = cv2.resize(digit_region, (28, 28), interpolation=cv2.INTER_AREA)
        # Normalize the resized digit
        normalized_digit = resized_digit / 255.0
        # Reshape the digit for prediction
        reshaped_digit = normalized_digit.reshape(1, 28, 28, 1)
        return reshaped_digit
    else:
        return None


# Streamlit app
st.title("Real-Time Hand Written Digit Recognition")
st.divider()
st.header('Instructions:')
st.write('''1. For best results use drawable canvas on your phone with black background and white font colour. (This 
is because the dataset used to train the model had aforementioned colour combination and grayscaling wasn't applied.) 
2. Prevent glaring and dimming in video capture.
3. Center the number in between the box present in the video capture.
4. Use landscape mode to draw so that the background in the frame remains covered with black canvas.'''
         )
st.divider()
# Placeholder for the video feed
video_placeholder = st.empty()

# Start video capture
cap = cv2.VideoCapture(0)

# Create a unique key for the stop button
stop_button_key = "stop_button"
stop_button_pressed = st.button("Stop", key=stop_button_key)

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video")
        break

    # Define a region of interest (ROI) for digit recognition
    x, y, w, h = 300, 100, 200, 200
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi = frame[y:y + h, x:x + w]

    # Preprocess the ROI for prediction
    processed_roi = preprocess(roi)
    if processed_roi is not None:
        # Predict the digit
        prediction = model.predict(processed_roi)
        digit = np.argmax(prediction)
        # Display the prediction on the video feed within the ROI
        cv2.putText(roi, f'Digit: {digit}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame in the Streamlit app
    video_placeholder.image(frame, channels="BGR")

    # Check if the stop button is pressed

    if stop_button_pressed:
        break

cap.release()
