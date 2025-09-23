import streamlit as st
import cv2
import numpy as np
import tempfile
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
import os
import requests
from io import BytesIO
from PIL import Image

# URL of the trained model weights.
WEIGHTS_URL = "https://drive.google.com/file/d/1fN1r5vyefdF-wsGdrfEA3oQcTz1xtg7_/view?usp=sharing"

# Model Building and Loading Function 

@st.cache_resource
def load_trained_model():
    """
    Builds the model architecture and loads the pre-trained weights.
    This function is cached to run only once.
    """
    # Define the model architecture
    model = keras.Sequential([
        keras.Input(shape=(256, 256, 1)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(2, (3, 3), activation='tanh', padding='same')
    ], name='colorizer')

    # Download weights
    try:
        # Check if the file already exists to avoid re-downloading
        if not os.path.exists('colorizer_weights.h5'):
            st.info("Downloading trained model weights...")
            response = requests.get(WEIGHTS_URL, allow_redirects=True, stream=True)
            response.raise_for_status() # Raise an error for bad status codes

            # Use a progress bar for large files
            total_size = int(response.headers.get('content-length', 0))
            bytes_downloaded = 0
            progress_bar = st.progress(0)
            
            with open('colorizer_weights.h5', 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        progress = min(bytes_downloaded / total_size, 1.0)
                        progress_bar.progress(progress)
            
            progress_bar.empty()
            st.success("Weights downloaded successfully!")
        
        # Load weights
        model.load_weights('colorizer_weights.h5')
        st.success("Model loaded with trained weights!")
    except Exception as e:
        st.warning(f"Could not download or load weights from the specified URL. Running with untrained model. Error: {e}")

    return model

def colorize_frame_tf(model, frame):
    """
    Colorizes a single frame using the TensorFlow model.
    """
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab_frame[:, :, 0]
    
    # Pre-process the frame for the model
    scaled_l = cv2.resize(l_channel, (256, 256))
    scaled_l = scaled_l.astype("float32") / 255.0
    scaled_l = np.expand_dims(scaled_l, axis=-1)
    scaled_l = np.expand_dims(scaled_l, axis=0)
    
    # Predict the color channels
    ab_predicted = model.predict(scaled_l, verbose=0)[0]
    
    # Resize the predicted color channels back to the original frame size
    ab_predicted = cv2.resize(ab_predicted, (frame.shape[1], frame.shape[0]))
    
    # Combine the original L channel with the predicted ab channels
    colorized_lab = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    colorized_lab[:, :, 0] = l_channel
    colorized_lab[:, :, 1:] = ab_predicted * 127 + 128
    
    # Convert from LAB to BGR for display
    colorized_frame = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
    
    return colorized_frame

# Streamlit UI and Logic

st.title("TensorFlow Video Colorizer")
st.markdown("Upload a video to see it colorized in real-time using a TensorFlow model.")

# Load the model with trained weights
model = load_trained_model()

# File uploader widget
uploaded_file = st.file_uploader("Choose a video file...", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    try:
        # Use a temporary file to save the uploaded video
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        
        # Open the video file with OpenCV
        cap = cv2.VideoCapture(tfile.name)
        
        if not cap.isOpened():
            st.error("Error: Could not open the video file. Please check the format.")
        else:
            st.subheader("Colorized Video")
            video_placeholder = st.empty()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply the colorization model
                processed_frame = colorize_frame_tf(model, frame)
                
                # Convert color space for Streamlit display
                processed_frame = cv2.cvtColor(processed_frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
                
                # Display the processed frame
                video_placeholder.image(processed_frame, use_column_width=True)
            
            st.success("Video processing complete!")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please ensure your video file is not corrupted and is in a supported format.")

    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'tfile' in locals():
            os.unlink(tfile.name)
else:
    st.info("Upload a video to get started.")
