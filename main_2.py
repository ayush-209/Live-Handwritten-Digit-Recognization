import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

model = keras.models.load_model('mnist_model_4.keras')


#%% preprocess the image for prediction
def preprocess(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            digit_region = gray[y:y + h, x:x + w]
            resized_digit = cv2.resize(digit_region, (28, 28), interpolation=cv2.INTER_AREA)
            normalized_digit = resized_digit / 255.0
            reshaped_digit = normalized_digit.reshape(1, 28, 28, 1)
            return reshaped_digit
        else:
            return None
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return None


#%% VideoTransformer class (processing video frames)
class DigitRecognitionTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        try:
            image = frame.to_ndarray(format="bgr24")
            x, y, w, h = 300, 100, 200, 200
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi = image[y:y + h, x:x + w]
            processed_roi = preprocess(roi)
            if processed_roi is not None:
                prediction = self.model.predict(processed_roi)
                digit = np.argmax(prediction)
                cv2.putText(roi, f'Digit: {digit}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return image
        except Exception as e:
            st.error(f"Error in transform: {e}")
            return frame


#%% streamlit front-end part
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

webrtc_ctx = webrtc_streamer(
    key="example",
    video_processor_factory=DigitRecognitionTransformer,
    async_transform=True,
    media_stream_constraints={"video": True, "audio": False},
)

if webrtc_ctx.video_processor:
    st.button("Stop")
