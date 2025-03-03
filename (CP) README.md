# Face Feature Detection Application

A Streamlit-based web application for real-time face feature detection using OpenCV's Haar cascade classifiers. This application provides an interactive interface for detecting and analyzing facial features in images.

## ⚡ Quick Start Guide

### Important: Running the Application

❗ **DO NOT run the Python script directly**. Always use the Streamlit CLI:

```powershell
# CORRECT WAY ✅
& ".venv/Scripts/streamlit.exe" "run" "(CP) Streamlit Project/(CP) streamlit_app.py"

# WRONG WAY ❌
python "(CP) Streamlit Project/(CP) streamlit_app.py"  # Don't do this
```

### Step-by-Step Setup

1. **Activate Virtual Environment**
   ```powershell
   .venv\Scripts\activate
   ```

2. **Install Dependencies** (if not already installed)
   ```powershell
   pip install streamlit opencv-python numpy pandas plotly pillow
   ```

3. **Run the Application**
   ```powershell
   # From project root (CAI2840C/):
   & ".venv/Scripts/streamlit.exe" "run" "(CP) Streamlit Project/(CP) streamlit_app.py"
   ```

4. **Access the Application**
   - Open your browser
   - Go to http://localhost:8501
   - The app will automatically download required model files on first run

### Common Issues

If you see warnings about "missing ScriptRunContext":
- ✋ Stop the application
- ✅ Make sure to use the Streamlit CLI command shown above
- ❌ Don't run the Python script directly

## 🎯 Features

### Core Detection Capabilities
- **Face Detection**: Identifies and locates faces in images
- **Eye Detection**: Detects eyes within identified face regions
- **Smile Detection**: Recognizes smiles within face regions
- **Real-time Processing**: Instant feature detection with adjustable parameters

### Interactive Interface
- **Feature Selection**
  - Toggle individual detection features (Face, Eyes, Smile)
  - Customize detection parameters for each feature
  - Real-time parameter adjustment

- **Detection Parameters**
  - Scale Factor (1.1 - 2.0)
    - Controls detection sensitivity
    - Lower values: More detections, slower performance
    - Higher values: Fewer detections, better performance
  - Minimum Neighbors (1 - 10)
    - Filters false positives
    - Higher values: More reliable detections
    - Lower values: More detections, including potential false positives

### Visualization Features
- Side-by-side display of original and processed images
- Customizable bounding box colors for each feature
- Optional confidence score display
- Processing time monitoring
- Interactive feature statistics with bar charts

### Export Options
- Download processed images with detections
- Export detection statistics as CSV
- Save and share results easily

## 🛠️ Technical Implementation

### Core Technologies
- **Streamlit**: Web application framework
- **OpenCV (cv2)**: Computer vision and image processing
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and statistics
- **Plotly**: Interactive visualizations
- **PIL**: Image processing
- **Base64**: Image encoding for downloads

### Key Components

#### Cascade Classifiers
The application uses three Haar cascade classifiers:
- `haarcascade_frontalface_default.xml`: Primary face detection
- `haarcascade_eye.xml`: Eye detection within face regions
- `haarcascade_smile.xml`: Smile detection within face regions

#### Detection Pipeline
1. **Image Preprocessing**
   - Convert to grayscale for efficient processing
   - Scale and normalize as needed

2. **Face Detection**
   - Apply primary cascade classifier
   - Define regions of interest (ROIs)
   - Draw bounding boxes

3. **Feature Detection**
   - Process each face ROI independently
   - Apply eye and smile classifiers
   - Calculate confidence scores

4. **Visualization**
   - Draw color-coded bounding boxes
   - Update statistics
   - Generate interactive charts

## 📝 Usage Instructions

1. **Starting the Application**
   ```powershell
   # From project root:
   & ".venv/Scripts/streamlit.exe" "run" "(CP) Streamlit Project/(CP) streamlit_app.py"
   ```

2. **Using the Interface**
   - Upload an image using the file uploader
   - Select features to detect from the sidebar
   - Adjust detection parameters as needed
   - View results in real-time
   - Download processed images or statistics

3. **Optimizing Detection**
   - Start with default parameters
   - Adjust scale factor for detection accuracy
   - Modify minimum neighbors to reduce false positives
   - Use the confidence display for validation

## ⚙️ Requirements

- Python 3.6+
- OpenCV (cv2)
- Streamlit
- NumPy
- Pandas
- Plotly Express
- Pillow (PIL)

## 🚀 Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create and Activate Virtual Environment**
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install streamlit opencv-python numpy pandas plotly pillow
   ```

## 🌟 Enhancements

### 1. Automatic Model Management
- Automatic download of cascade files
- Validation of model integrity
- Error handling for missing/corrupt models
- Proper model versioning

### 2. Interactive UI/UX
- Real-time parameter adjustment
- Custom color selection for visualizations
- Multiple visualization options
- Intuitive sidebar controls
- Responsive layout

### 3. Analytics Features
- Detection statistics tracking
- Performance monitoring
- Interactive data visualization
- Export capabilities

### 4. Error Handling
- Graceful handling of missing models
- Input validation
- Clear error messages
- Recovery mechanisms

## 🔄 Future Improvements

1. **Detection Capabilities**
   - Additional facial feature detection
   - Emotion recognition integration
   - Age and gender estimation
   - Multi-face tracking

2. **Performance Optimization**
   - GPU acceleration support
   - Batch processing
   - Image preprocessing options
   - Caching mechanisms

3. **UI Enhancements**
   - Camera input support
   - Batch upload processing
   - Advanced visualization options
   - Custom detection zones

4. **Analytics**
   - Advanced statistics
   - Time-series analysis
   - Detection confidence metrics
   - Export formats

## 🔧 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check internet connection for downloads
   - Verify models directory permissions
   - Ensure proper file paths
   - Validate model integrity

2. **Detection Issues**
   - Adjust scale factors for better results
   - Modify minimum neighbors
   - Check image quality and lighting
   - Verify feature selection

3. **Performance Issues**
   - Reduce image size
   - Optimize detection parameters
   - Limit active features
   - Check system resources

## 📁 Project Structure
```
project_root/ (CAI2840C/)
├── .venv/                          # Virtual environment
├── (CP) Streamlit Project/         # Application directory
│   ├── models/                     # Cascade classifier files
│   └── (CP) streamlit_app.py      # Main application
└── README.md                       # Documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenCV for providing the Haar cascade classifiers
- Streamlit for the excellent web application framework
- The computer vision community for their valuable resources and tutorials 