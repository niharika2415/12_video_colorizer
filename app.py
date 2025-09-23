import streamlit as st
import cv2
import numpy as np
import tempfile

# Colorization Functions

def colorize_sepia(grayscale_frame):
    """
    Applies a sepia tone effect to a grayscale frame.
    This is a simple matrix multiplication that simulates an older camera tone.
    """
    # Create a 3-channel sepia matrix
    sepia_matrix = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ], dtype=np.float32)
    gray_3_channel = cv2.merge([grayscale_frame, grayscale_frame, grayscale_frame])
    sepia_frame = cv2.transform(gray_3_channel, sepia_matrix)
    return sepia_frame

def colorize_custom_hue(grayscale_frame):
    """
    Applies a custom blueish hue to a grayscale frame.
    This is a simple method that merges a grayscale image with a constant color.
    """
    # Create a 3-channel image with the grayscale frame in one channel and
    # a custom color in the others to create a tint.
    # We use a blue-ish tint in this example (B, G, R)
    tinted_frame = cv2.merge([grayscale_frame, grayscale_frame * 0.5, grayscale_frame * 0.1])
    return tinted_frame

# --- Streamlit UI and Logic ---

st.title("Real-time Video Colorizer")
st.markdown("Upload a video and choose a filter to see the effect in real-time.")

# Sidebar for filter selection
st.sidebar.title("Filter Selection")
filter_name = st.sidebar.radio(
    "Choose a colorization filter:",
    ('Original', 'Sepia', 'Custom Hue')
)

# File uploader widget
uploaded_file = st.file_uploader("Upload a video file (MP4)", type=['mp4'])

if uploaded_file is not None:
    # Use a temporary file to save the uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Open the video file with OpenCV
    cap = cv2.VideoCapture(tfile.name)
    
    # Check if the video file was opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open the video file.")
    else:
        # Create a placeholder to display the video frames
        video_placeholder = st.empty()
        
        # Loop through the video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to grayscale for processing
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply the selected filter
            if filter_name == 'Sepia':
                processed_frame = colorize_sepia(gray_frame)
            elif filter_name == 'Custom Hue':
                processed_frame = colorize_custom_hue(gray_frame)
            else:
                processed_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            
            # Convert color space for Streamlit display
            processed_frame = cv2.cvtColor(processed_frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
            
            # Display the processed frame
            video_placeholder.image(processed_frame, use_column_width=True)

    # Release video capture and close temporary file
    cap.release()
    tfile.close()

st.info("The application will process video frames after you upload a file.")
