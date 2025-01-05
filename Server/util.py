import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
import joblib

# Function to print the file path to verify it's correct
def print_file_paths():
    face_cascade_path = r'C:\Users\SAKET\OneDrive\Desktop\AIML\Sports Person Classifier\Server\opencv\haarcascade\haarcascade_frontalface_default.xml'
    eye_cascade_path = r'C:\Users\SAKET\OneDrive\Desktop\AIML\Sports Person Classifier\Server\opencv\haarcascade\haarcascade_eye.xml'
    
    print("Face Cascade Path: ", face_cascade_path)
    print("Eye Cascade Path: ", eye_cascade_path)

# Function to load the Haar Cascade files and check if they are loaded properly
def load_haar_cascades():
    face_cascade_path = r'C:\Users\SAKET\OneDrive\Desktop\AIML\Sports Person Classifier\Server\opencv\haarcascade\haarcascade_frontalface_default.xml'
    eye_cascade_path = r'C:\Users\SAKET\OneDrive\Desktop\AIML\Sports Person Classifier\Server\opencv\haarcascade\haarcascade_eye.xml'

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    if face_cascade.empty():
        print("Error loading face cascade")
    else:
        print("Face cascade loaded successfully")

    if eye_cascade.empty():
        print("Error loading eye cascade")
    else:
        print("Eye cascade loaded successfully")

    return face_cascade, eye_cascade

# Function to get a base64 encoded image
def get_b64_test_image_for_virat():
    # Return a valid base64 image for testing
    with open("path_to_image_of_virat.jpg", "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode('utf-8')
    return b64_string

# Function to crop the image if 2 eyes are detected
def get_cropped_image_if_2_eyes(file_path, image_base64_data):
    # Load the Haar cascades
    face_cascade, eye_cascade = load_haar_cascades()
    
    # Decode the image from base64
    image_data = base64.b64decode(image_base64_data)
    npimg = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        raise ValueError("No face detected!")

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            # Crop the image to the face with two eyes detected
            cropped_img = roi_color
            return cropped_img

    raise ValueError("Could not detect two eyes!")

# Function to classify the image (dummy classification for illustration)
def classify_image(image_base64_data, file_path=None):
    print_file_paths()  # Debug: print paths of the Haar cascades
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    # Here you can integrate your classification model for predicting the sportsperson
    # For demonstration, returning a dummy classification result
    return "Virat Kohli"

# Main execution block
if __name__ == "__main__":
    try:
        print("Loading saved artifacts...start")
        # Load the classifier (replace with your actual classifier loading code)
        # Example: clf = joblib.load('model.pkl')

        print("Loading saved artifacts...done")

        # Test the classification function with a sample image
        result = classify_image(get_b64_test_image_for_virat(), None)
        print(f"Classification Result: {result}")

    except Exception as e:
        print(f"Error: {e}")
