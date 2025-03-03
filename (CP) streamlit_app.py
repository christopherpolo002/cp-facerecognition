import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import time
import pandas as pd
import plotly.express as px
import os

# Page configuration
st.set_page_config(layout="wide", page_title="(CP)Face Feature Detection")

# Initialize session state variables
if 'face_cascade' not in st.session_state:
    st.session_state.face_cascade = None
if 'eye_cascade' not in st.session_state:
    st.session_state.eye_cascade = None
if 'smile_cascade' not in st.session_state:
    st.session_state.smile_cascade = None

# Load the cascade classifiers
def load_cascades():
    try:
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, 'models')
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Define model files
        models = {
            'face': 'haarcascade_frontalface_default.xml',
            'eye': 'haarcascade_eye.xml',
            'smile': 'haarcascade_smile.xml'
        }
        
        # Download missing models
        base_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
        for model_name, model_file in models.items():
            model_path = os.path.join(models_dir, model_file)
            if not os.path.exists(model_path):
                import urllib.request
                url = base_url + model_file
                urllib.request.urlretrieve(url, model_path)
                st.info(f"Downloaded {model_file}")
        
        # Load cascade classifiers
        face_cascade = cv2.CascadeClassifier(os.path.join(models_dir, models['face']))
        eye_cascade = cv2.CascadeClassifier(os.path.join(models_dir, models['eye']))
        smile_cascade = cv2.CascadeClassifier(os.path.join(models_dir, models['smile']))
        
        if face_cascade.empty():
            st.error("Error: Could not load face cascade classifier.")
            return None, None, None
        if eye_cascade.empty():
            st.error("Error: Could not load eye cascade classifier.")
            return None, None, None
        if smile_cascade.empty():
            st.error("Error: Could not load smile cascade classifier.")
            return None, None, None
            
        return face_cascade, eye_cascade, smile_cascade
    except Exception as e:
        st.error(f"Error loading cascade classifiers: {str(e)}")
        return None, None, None

# Load cascades at startup
st.session_state.face_cascade, st.session_state.eye_cascade, st.session_state.smile_cascade = load_cascades()

if not all([st.session_state.face_cascade, st.session_state.eye_cascade, st.session_state.smile_cascade]):
    st.error("Failed to load one or more cascade classifiers. Please check that the model files exist in the 'models' directory.")
    st.stop()

# Sidebar for detection settings
st.sidebar.title("Detection Settings")

# Feature selection
selected_features = st.sidebar.multiselect(
    "Select Detection Features",
    ["Face", "Eyes", "Smile"],
    default=["Face"]
)

# Detection parameters
face_scale = st.sidebar.slider("Face Detection Scale Factor", 1.1, 2.0, 1.3, 0.1)
face_neighbors = st.sidebar.slider("Face Detection Min Neighbors", 1, 10, 5)

eye_scale = st.sidebar.slider("Eye Detection Scale Factor", 1.1, 2.0, 1.3, 0.1)
eye_neighbors = st.sidebar.slider("Eye Detection Min Neighbors", 1, 10, 5)

smile_scale = st.sidebar.slider("Smile Detection Scale Factor", 1.1, 2.0, 1.3, 0.1)
smile_neighbors = st.sidebar.slider("Smile Detection Min Neighbors", 1, 10, 5)

# Visualization options
show_confidence = st.sidebar.checkbox("Show Confidence Scores", True)
show_processing_time = st.sidebar.checkbox("Show Processing Time", True)
show_stats = st.sidebar.checkbox("Show Feature Statistics", True)

# Color settings
face_color = st.sidebar.color_picker("Face Detection Color", "#0000FF")
eye_color = st.sidebar.color_picker("Eye Detection Color", "#00FF00")
smile_color = st.sidebar.color_picker("Smile Detection Color", "#FF0000")

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Function to detect features in an image
def detect_features(image):
    start_time = time.time()
    stats = {"faces": 0, "eyes": 0, "smiles": 0}
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    if "Face" in selected_features:
        faces = st.session_state.face_cascade.detectMultiScale(
            gray, scaleFactor=face_scale, minNeighbors=face_neighbors
        )
        stats["faces"] = len(faces)
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), hex_to_rgb(face_color), 2)
            face_roi_gray = gray[y:y+h, x:x+w]
            face_roi_color = image[y:y+h, x:x+w]
            
            # Detect eyes
            if "Eyes" in selected_features:
                eyes = st.session_state.eye_cascade.detectMultiScale(
                    face_roi_gray, scaleFactor=eye_scale, minNeighbors=eye_neighbors
                )
                stats["eyes"] += len(eyes)
                
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), hex_to_rgb(eye_color), 2)
            
            # Detect smiles
            if "Smile" in selected_features:
                smiles = st.session_state.smile_cascade.detectMultiScale(
                    face_roi_gray, scaleFactor=smile_scale, minNeighbors=smile_neighbors
                )
                stats["smiles"] += len(smiles)
                
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(face_roi_color, (sx, sy), (sx+sw, sy+sh), hex_to_rgb(smile_color), 2)
    
    processing_time = time.time() - start_time
    return image, stats, processing_time

# Function to get image download link
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Main app
st.title("(CP) Face Feature Detection")

# File uploader
img_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    # Read image
    image = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), 1)
    
    # Create columns for display
    col1, col2 = st.columns(2)
    
    # Display original image
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Process and display detected features
    processed_image, stats, proc_time = detect_features(image.copy())
    
    with col2:
        st.subheader("Processed Image")
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        
        if show_processing_time:
            st.write(f"Processing Time: {proc_time:.3f} seconds")
    
    # Display statistics
    if show_stats and stats["faces"] > 0:
        st.subheader("Detection Statistics")
        df = pd.DataFrame({
            'Feature': ['Faces', 'Eyes', 'Smiles'],
            'Count': [stats['faces'], stats['eyes'], stats['smiles']]
        })
        
        fig = px.bar(df, x='Feature', y='Count', title='Detected Features Count')
        st.plotly_chart(fig)
    
    # Download options
    st.subheader("Download Options")
    
    # Convert processed image to PIL format
    processed_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    
    # Create download link for processed image
    st.markdown(
        get_image_download_link(processed_pil, "processed_image.jpg", "Download Processed Image"),
        unsafe_allow_html=True
    )
    
    # Export statistics
    if show_stats:
        stats_df = pd.DataFrame([stats])
        csv = stats_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="detection_stats.csv">Download Statistics (CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)
