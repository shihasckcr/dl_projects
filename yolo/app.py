from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import time
import cv2
from werkzeug.utils import secure_filename
from flask_cors import CORS
from yolo_model import predict_yolo

app = Flask(__name__, static_folder="static", template_folder="templates")  
CORS(app, supports_credentials=True, expose_headers=["Content-Type", "Authorization", "X-Custom-Header"])

# Configurations
BASE_DIR = "/home/shihas/Documents/DL/yolo"
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, "uploaded")
app.config['RESULTS_FOLDER'] = os.path.join(BASE_DIR, "static/results")
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    """Serve the frontend index.html"""
    return render_template('index.html')  # Ensure 'index.html' is inside the 'templates' folder

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/static/results/<filename>')
def serve_result_image(filename):
    """Serve saved result images."""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and return YOLO predictions."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform YOLO prediction
        predictions, img = predict_yolo(file_path)

        # Save result image
        result_image_name = f"result_{filename}"
        result_image_path = os.path.join(app.config['RESULTS_FOLDER'], result_image_name)
        cv2.imwrite(result_image_path, img)

        # Ensure the image is fully saved before responding
        time.sleep(1)
        while not os.path.exists(result_image_path):
            time.sleep(0.5)

        if os.path.exists(result_image_path):
            print("✅ Image successfully saved:", result_image_path)
        else:
            print("❌ Error: Image NOT saved!")

        image_url = f"http://127.0.0.1:5000/static/results/{result_image_name}"
        print("Generated image URL:", image_url)

        # Return only labels in predictions
        response = jsonify({
            "predictions": [pred["label"] for pred in predictions],  
            "result_image": image_url
        })
        response.headers["X-Custom-Header"] = "YOLOPrediction"
        return response

    return jsonify({"error": "Invalid file format"}), 400

@app.after_request
def add_cors_headers(response):
    """Add necessary CORS headers."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Expose-Headers"] = "Content-Type, Authorization, X-Custom-Header"
    response.headers["X-Custom-Header"] = "YOLO-Test"
    return response

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
