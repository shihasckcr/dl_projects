from flask import Flask, request, jsonify, render_template
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from flask_cors import CORS


app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  

app.config['UPLOAD_FOLDER'] = "uploaded"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


model_path = "/home/shihas/dl_projects/Crop Disease Detection/crop_disease_model.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = tf.keras.models.load_model(model_path)


class_names = [
    "Strawberry___Leaf_scorch", "Apple___healthy", "Corn___Common_rust", "Grape___healthy", 
    "Tomato___Bacterial_spot", "Soybean___healthy", "Tomato___Leaf_Mold", "Pepper,_bell___Bacterial_spot", 
    "Corn___Northern_Leaf_Blight", "Tomato___Early_blight", "Apple___Apple_scab", "Potato___Early_blight", 
    "Raspberry___healthy", "Pepper,_bell___healthy", "Tomato___Septoria_leaf_spot", "Orange___Haunglongbing_(Citrus_greening)",
    "Cherry___Powdery_mildew", "Tomato___Spider_mites Two-spotted_spider_mite", "Grape___Black_rot", "Apple___Black_rot",
    "Background_without_leaves", "Peach___healthy", "Grape___Esca_(Black_Measles)", "Tomato___Target_Spot",
    "Blueberry___healthy", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Potato___Late_blight", "Squash___Powdery_mildew",
    "Tomato___healthy", "Tomato___Tomato_mosaic_virus", "Strawberry___healthy", "Peach___Bacterial_spot", 
    "Tomato___Late_blight", "Corn___healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Cherry___healthy", 
    "Potato___healthy", "Apple___Cedar_apple_rust", "Corn___Cercospora_leaf_spot Gray_leaf_spot"
]

def process_image(image_path):
    """Preprocess image for model prediction."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    """Serve the frontend page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload and prediction."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    print(f"✅ File uploaded successfully: {filename}")

    try:
        
        image = process_image(file_path)
        prediction = model.predict(image)
        predicted_class = class_names[np.argmax(prediction)]
        print(f"🔍 Predicted class: {predicted_class}")

        return jsonify({"prediction": predicted_class})
    
    except Exception as e:
        print(f"❌ Prediction Error: {str(e)}")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
