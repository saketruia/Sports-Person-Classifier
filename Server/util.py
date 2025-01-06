import cv2
import numpy as np
import base64
import joblib
import json
from wavelet import w2d

__model = None  # Add this line to declare the variable
__class_names = None
__class_name_to_number = {}
__class_number_to_name = {}

# Function to print the file path to verify it's correct
def print_file_paths():
    face_cascade_path = r'C:/Users/SAKET/OneDrive/Desktop/AIML/Sports Person Classifier/Server/opencv/haarcascade/haarcascade_frontalface_default.xml'
    eye_cascade_path = r'C:/Users/SAKET/OneDrive/Desktop/AIML/Sports Person Classifier/Server/opencv/haarcascade/haarcascade_eye.xml'
    
    print("Face Cascade Path: ", face_cascade_path)
    print("Eye Cascade Path: ", eye_cascade_path)

# Function to load the Haar Cascade files and check if they are loaded properly
def load_haar_cascades():
    face_cascade_path = r'C:/Users/SAKET/OneDrive/Desktop/AIML/Sports Person Classifier/Server/opencv/haarcascade/haarcascade_frontalface_default.xml'
    eye_cascade_path = r'C:/Users/SAKET/OneDrive/Desktop/AIML/Sports Person Classifier/Server/opencv/haarcascade/haarcascade_eye.xml'

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
    with open("./test_images/virat1.jpg", "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode('utf-8')
    return b64_string

# Function to crop the image if 2 eyes are detected
def get_cropped_image_if_2_eyes(image_base64_data):
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
def classify_image(image_base64_data):
    print_file_paths()  # Debug: print paths of the Haar cascades
    img = get_cropped_image_if_2_eyes(image_base64_data)
    
    # Process image for classification
    scalled_raw_img = cv2.resize(img, (32, 32))
    img_har = w2d(img, 'db1', 5)
    scalled_img_har = cv2.resize(img_har, (32, 32))
    
    # Combine raw and wavelet images
    combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

    len_image_array = 32*32*3 + 32*32

    final = combined_img.reshape(1, len_image_array).astype(float)

    # Get predicted class from model
    predicted_class_number = __model.predict(final)[0]
    predicted_class = __class_number_to_name[predicted_class_number]
    
    result = {
        'class': predicted_class,
        'class_probability': np.around(__model.predict_proba(final) * 100, 2).tolist()[0],
        'class_dictionary': __class_name_to_number
    }

    return result

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('./artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")

# Main execution block
if __name__ == "__main__":
    load_saved_artifacts()
    print(classify_image(get_b64_test_image_for_virat()))
