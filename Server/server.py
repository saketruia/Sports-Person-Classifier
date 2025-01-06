from flask import Flask, request, jsonify
import util
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This handles CORS for all routes

@app.route('/classify_image', methods=['POST'])
def classify_image():
    # Assuming the image is sent as a base64-encoded string in a JSON payload
    image_data = request.json.get('image_data')

    if not image_data:
        return jsonify({'error': 'No image data provided'}), 400

    try:
        # Process the image and classify it
        result = util.classify_image(image_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    util.load_saved_artifacts()  # Load model and other necessary artifacts
    app.run(port=5000)
