import urllib.request
import os

def download_models():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # URLs for the model files
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt"
    model_url = "https://github.com/opencv/opencv_3rdparty/raw/19512576c112aa2c7b6328cb0e8d589a4a90a26d/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    
    # Download files
    print("Downloading model files...")
    urllib.request.urlretrieve(prototxt_url, 'models/deploy.prototxt')
    print("Downloaded deploy.prototxt")
    urllib.request.urlretrieve(model_url, 'models/res10_300x300_ssd_iter_140000_fp16.caffemodel')
    print("Downloaded caffemodel")

    # URLs for the cascade files
    cascade_files = {
        'haarcascade_frontalface_default.xml': "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
        'haarcascade_eye.xml': "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml",
        'haarcascade_smile.xml': "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_smile.xml"
    }
    
    # Download files
    print("Downloading cascade files...")
    for filename, url in cascade_files.items():
        filepath = os.path.join('models', filename)
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded {filepath}")

    print("Download complete!")

if __name__ == "__main__":
    download_models() 