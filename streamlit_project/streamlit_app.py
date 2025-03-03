import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import time
import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import os

# Page configuration
st.set_page_config(layout="wide", page_title="Face Detection and Comparison")

# Initialize session state for caching results
if 'face_cascade' not in st.session_state:
    st.session_state.face_cascade = None
if 'eye_cascade' not in st.session_state:
    st.session_state.eye_cascade = None
if 'smile_cascade' not in st.session_state:
    st.session_state.smile_cascade = None

# Sidebar configuration
st.sidebar.title("Detection Settings")

# App mode selection
app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    ["Single Image Analysis", "Face Comparison"]
)

# Feature selection in sidebar
features = st.sidebar.multiselect(
    'Select features to detect',
    ['Face', 'Eyes', 'Smile'],
    ['Face']
)

# Detection parameters in sidebar
detection_params = {}
if 'Face' in features:
    st.sidebar.subheader("Face Detection Parameters")
    detection_params['face_scale'] = st.sidebar.slider("Face Scale Factor", 1.1, 2.0, 1.3, 0.1)
    detection_params['face_neighbors'] = st.sidebar.slider("Face Min Neighbors", 1, 10, 5)

if 'Eyes' in features:
    st.sidebar.subheader("Eye Detection Parameters")
    detection_params['eye_scale'] = st.sidebar.slider("Eye Scale Factor", 1.1, 2.0, 1.3, 0.1)
    detection_params['eye_neighbors'] = st.sidebar.slider("Eye Min Neighbors", 1, 10, 5)

if 'Smile' in features:
    st.sidebar.subheader("Smile Detection Parameters")
    detection_params['smile_scale'] = st.sidebar.slider("Smile Scale Factor", 1.1, 2.0, 1.7, 0.1)
    detection_params['smile_neighbors'] = st.sidebar.slider("Smile Min Neighbors", 1, 30, 20)

# Visualization options in sidebar
st.sidebar.subheader("Visualization Options")
show_confidence = st.sidebar.checkbox("Show Confidence Scores", True)
show_processing_time = st.sidebar.checkbox("Show Processing Time", True)
show_feature_stats = st.sidebar.checkbox("Show Feature Statistics", True)

# Color picker for bounding boxes
st.sidebar.subheader("Bounding Box Colors")
face_color = st.sidebar.color_picker("Face Detection Color", "#FF0000")
eye_color = st.sidebar.color_picker("Eye Detection Color", "#00FF00")
smile_color = st.sidebar.color_picker("Smile Detection Color", "#0000FF")

def process_uploaded_image(uploaded_file):
    # Read the file and convert it to opencv Image
    raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

def extract_face_encoding(face_img):
    # Convert to RGB (face_recognition expects RGB)
    rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    # Get face encoding using face_recognition
    import face_recognition
    encodings = face_recognition.face_encodings(rgb_img)
    return encodings[0] if len(encodings) > 0 else None

def compare_faces(face1_encoding, face2_encoding):
    if face1_encoding is None or face2_encoding is None:
        return 0.0
    # Calculate similarity using cosine similarity
    similarity = cosine_similarity(
        face1_encoding.reshape(1, -1),
        face2_encoding.reshape(1, -1)
    )[0][0]
    return similarity

# Load the cascade classifiers
@st.cache_resource
def load_cascades():
    try:
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct paths to cascade files
        face_cascade = cv2.CascadeClassifier(os.path.join(script_dir, 'models', 'haarcascade_frontalface_default.xml'))
        eye_cascade = cv2.CascadeClassifier(os.path.join(script_dir, 'models', 'haarcascade_eye.xml'))
        smile_cascade = cv2.CascadeClassifier(os.path.join(script_dir, 'models', 'haarcascade_smile.xml'))
        
        # Verify that the classifiers loaded successfully
        if face_cascade.empty() or eye_cascade.empty() or smile_cascade.empty():
            st.error("Error: Could not load one or more cascade classifiers. Please check that the model files exist in the 'models' directory.")
            return None, None, None
            
        return face_cascade, eye_cascade, smile_cascade
    except Exception as e:
        st.error(f"Error loading cascade classifiers: {str(e)}")
        return None, None, None

# Load cascades at startup
face_cascade, eye_cascade, smile_cascade = load_cascades()

if face_cascade is None or eye_cascade is None or smile_cascade is None:
    st.error("Failed to load cascade classifiers. Please check that the model files exist in the 'models' directory.")
    st.stop()

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))[::-1]

# Function for detecting facial features with confidence scores
def detect_features(image, face_cascade, eye_cascade, smile_cascade, params):
    start_time = time.time()
    results = {
        'faces': [], 'eyes': [], 'smiles': [],
        'processing_time': 0,
        'statistics': {}
    }
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    if face_cascade:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=params.get('face_scale', 1.3),
            minNeighbors=params.get('face_neighbors', 5),
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        results['faces'] = faces.tolist() if len(faces) > 0 else []
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(image, (x, y), (x+w, y+h), hex_to_rgb(face_color), 2)
            if show_confidence:
                confidence = f"Face"
                cv2.putText(image, confidence, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hex_to_rgb(face_color), 2)
            
            # Region of interest for eyes and smile
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            
            # Detect eyes
            if eye_cascade:
                eyes = eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=params.get('eye_scale', 1.3),
                    minNeighbors=params.get('eye_neighbors', 5)
                )
                results['eyes'].extend([(ex+x, ey+y, ew, eh) for (ex, ey, ew, eh) in eyes])
                
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), hex_to_rgb(eye_color), 2)
                    if show_confidence:
                        cv2.putText(roi_color, "Eye", (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hex_to_rgb(eye_color), 2)
            
            # Detect smile
            if smile_cascade:
                smiles = smile_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=params.get('smile_scale', 1.7),
                    minNeighbors=params.get('smile_neighbors', 20)
                )
                results['smiles'].extend([(sx+x, sy+y, sw, sh) for (sx, sy, sw, sh) in smiles])
                
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), hex_to_rgb(smile_color), 2)
                    if show_confidence:
                        cv2.putText(roi_color, "Smile", (sx, sy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hex_to_rgb(smile_color), 2)
    
    # Calculate statistics
    results['statistics'] = {
        'total_faces': len(results['faces']),
        'total_eyes': len(results['eyes']),
        'total_smiles': len(results['smiles']),
        'eyes_per_face': len(results['eyes']) / len(results['faces']) if len(results['faces']) > 0 else 0,
        'smiles_per_face': len(results['smiles']) / len(results['faces']) if len(results['faces']) > 0 else 0
    }
    
    results['processing_time'] = time.time() - start_time
    return image, results

# Function to generate a download link for output file
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Main content based on app mode
if app_mode == "Single Image Analysis":
    st.title("Single Image Analysis")
    img_file_buffer = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    if img_file_buffer is not None:
        # Process single image
        image = process_uploaded_image(img_file_buffer)
        
        # Create placeholders to display input and output images
        col1, col2 = st.columns(2)
        
        # Display Input image in the first placeholder
        with col1:
            st.subheader("Input Image")
            st.image(image, channels='BGR')
        
        # Process image based on selected features
        processed_image = image.copy()
        processed_image, results = detect_features(
            processed_image,
            face_cascade if 'Face' in features else None,
            eye_cascade if 'Eyes' in features else None,
            smile_cascade if 'Smile' in features else None,
            detection_params
        )
        
        # Display processed image and results
        with col2:
            st.subheader("Processed Image")
            st.image(processed_image, channels='BGR')
            
            if show_processing_time:
                st.info(f"Processing Time: {results['processing_time']:.3f} seconds")
        
        # Display statistics and visualizations
        if show_feature_stats and results['statistics']:
            st.subheader("Detection Statistics")
            
            # Create three columns for statistics
            stat_cols = st.columns(3)
            
            with stat_cols[0]:
                st.metric("Total Faces", results['statistics']['total_faces'])
            with stat_cols[1]:
                st.metric("Total Eyes", results['statistics']['total_eyes'])
            with stat_cols[2]:
                st.metric("Total Smiles", results['statistics']['total_smiles'])
            
            # Create a bar chart of detections
            chart_data = pd.DataFrame({
                'Feature': ['Faces', 'Eyes', 'Smiles'],
                'Count': [
                    results['statistics']['total_faces'],
                    results['statistics']['total_eyes'],
                    results['statistics']['total_smiles']
                ]
            })
            
            fig = px.bar(
                chart_data,
                x='Feature',
                y='Count',
                title='Detection Counts by Feature',
                color='Feature'
            )
            st.plotly_chart(fig)
        
        # Export options
        st.subheader("Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            # Download processed image
            st.markdown(
                get_image_download_link(
                    Image.fromarray(processed_image[:, :, ::-1]),
                    "detected_features.jpg",
                    'Download Processed Image'
                ),
                unsafe_allow_html=True
            )
        
        with col2:
            # Download statistics as CSV
            if st.button("Download Statistics (CSV)"):
                df = pd.DataFrame([results['statistics']])
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="detection_statistics.csv">Download Statistics CSV</a>'
                st.markdown(href, unsafe_allow_html=True)

else:  # Face Comparison mode
    st.title("Face Comparison")
    
    # Create two columns for image upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("First Image")
        img1_file = st.file_uploader("Choose first image", type=['jpg', 'jpeg', 'png'], key="img1")
        
    with col2:
        st.subheader("Second Image")
        img2_file = st.file_uploader("Choose second image", type=['jpg', 'jpeg', 'png'], key="img2")
    
    if img1_file is not None and img2_file is not None:
        # Process both images
        image1 = process_uploaded_image(img1_file)
        image2 = process_uploaded_image(img2_file)
        
        # Create copies for processing
        processed_image1, results1 = detect_features(
            image1.copy(),
            face_cascade if 'Face' in features else None,
            eye_cascade if 'Eyes' in features else None,
            smile_cascade if 'Smile' in features else None,
            detection_params
        )
        
        processed_image2, results2 = detect_features(
            image2.copy(),
            face_cascade if 'Face' in features else None,
            eye_cascade if 'Eyes' in features else None,
            smile_cascade if 'Smile' in features else None,
            detection_params
        )
        
        # Display processed images side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(processed_image1, channels='BGR', caption="First Image")
            if show_processing_time:
                st.info(f"Processing Time: {results1['processing_time']:.3f} seconds")
        
        with col2:
            st.image(processed_image2, channels='BGR', caption="Second Image")
            if show_processing_time:
                st.info(f"Processing Time: {results2['processing_time']:.3f} seconds")
        
        # Face comparison
        if len(results1['faces']) > 0 and len(results2['faces']) > 0:
            st.subheader("Face Comparison Results")
            
            # Extract face regions
            x1, y1, w1, h1 = results1['faces'][0]  # Using first face from each image
            x2, y2, w2, h2 = results2['faces'][0]
            
            face1 = image1[y1:y1+h1, x1:x1+w1]
            face2 = image2[y2:y2+h2, x2:x2+w2]
            
            # Get face encodings and compare
            encoding1 = extract_face_encoding(face1)
            encoding2 = extract_face_encoding(face2)
            
            if encoding1 is not None and encoding2 is not None:
                similarity = compare_faces(encoding1, encoding2)
                
                # Display similarity score with a progress bar
                st.subheader("Face Similarity Score")
                st.progress(similarity)
                st.write(f"Similarity: {similarity:.2%}")
                
                # Create comparison metrics
                comparison_data = {
                    'Metric': ['Face Count', 'Eye Count', 'Smile Count'],
                    'Image 1': [
                        results1['statistics']['total_faces'],
                        results1['statistics']['total_eyes'],
                        results1['statistics']['total_smiles']
                    ],
                    'Image 2': [
                        results2['statistics']['total_faces'],
                        results2['statistics']['total_eyes'],
                        results2['statistics']['total_smiles']
                    ]
                }
                
                # Create and display comparison chart
                comparison_df = pd.DataFrame(comparison_data)
                fig = px.bar(
                    comparison_df,
                    x='Metric',
                    y=['Image 1', 'Image 2'],
                    title='Feature Comparison Between Images',
                    barmode='group'
                )
                st.plotly_chart(fig)
            else:
                st.error("Could not extract face features for comparison. Please ensure faces are clearly visible.")
        else:
            st.error("Could not detect faces in one or both images. Please ensure faces are clearly visible.")
        
        # Export options
        st.subheader("Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            # Download processed images
            st.markdown(
                get_image_download_link(
                    Image.fromarray(processed_image1[:, :, ::-1]),
                    "image1_processed.jpg",
                    'Download First Processed Image'
                ),
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                get_image_download_link(
                    Image.fromarray(processed_image2[:, :, ::-1]),
                    "image2_processed.jpg",
                    'Download Second Processed Image'
                ),
                unsafe_allow_html=True
            )
        
        # Export comparison results
        if st.button("Download Comparison Results (CSV)"):
            comparison_results = {
                'metric': ['similarity_score'] + list(results1['statistics'].keys()),
                'image1_value': [similarity] + list(results1['statistics'].values()),
                'image2_value': [similarity] + list(results2['statistics'].values())
            }
            df = pd.DataFrame(comparison_results)
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="comparison_results.csv">Download Comparison Results</a>'
            st.markdown(href, unsafe_allow_html=True)
