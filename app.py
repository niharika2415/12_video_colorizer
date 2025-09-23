import streamlit as st
import cv2
import numpy as np
import tempfile
import tensorflow as tf
from tensorflow import keras
import os
import gdown  # ✅ for downloading from Google Drive
from PIL import Image

# Google Drive File ID
FILE_ID = "1gZSAAAUDLLyGLN8Ylk7LrqvcjfF2L0Gl"  
MODEL_PATH = "colorizer_model.h5"

# Model Loading Function
@st.cache_resource
def load_trained_model():
    """
    Loads the pre-trained full model (.h5) from Google Drive.
    This function is cached to avoid re-downloading/reloading.
    """
    try:
        # Download model if not already present
        if not os.path.exists(MODEL_PATH):
            #st.info("Downloading trained model from Google Drive...")
            gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
            st.success("Model downloaded successfully!")

        # Load full model
        model = keras.models.load_model(MODEL_PATH)
        st.success("Model loaded with trained weights!")
        return model

    except Exception as e:
        st.error(f"Could not load trained model. Error: {e}")
        return None


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
    
    # Resize predicted channels back to original frame size
    ab_predicted = cv2.resize(ab_predicted, (frame.shape[1], frame.shape[0]))
    
    # Combine L channel with predicted ab channels
    colorized_lab = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    colorized_lab[:, :, 0] = l_channel
    colorized_lab[:, :, 1:] = ab_predicted * 127 + 128
    
    # Convert LAB → BGR
    colorized_frame = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
    
    return colorized_frame


# Streamlit UI and Logic

st.title("TensorFlow Video Colorizer")
st.markdown("Upload a video to see it colorized in real-time using a TensorFlow model.")

# Load the model
model = load_trained_model()

# File uploader widget
uploaded_file = st.file_uploader("Choose a video file...", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None and model is not None:
    try:
        # Save uploaded video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        
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
                
                # Apply model
                processed_frame = colorize_frame_tf(model, frame)
                
                # Convert for Streamlit display
                processed_frame = cv2.cvtColor(processed_frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
                
                video_placeholder.image(processed_frame, use_column_width=True)
            
            st.success("Video processing complete!")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please ensure your video file is valid and supported.")

    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'tfile' in locals():
            os.unlink(tfile.name)
else:
    st.info("Upload a video to get started.")
