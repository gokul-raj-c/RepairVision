# Import modules
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "car_vs_noncar_model")

#car_classifier = tf.keras.models.load_model("models/car_detection_model.h5")

car_classifier = tf.saved_model.load(MODEL_PATH)
infer = car_classifier.signatures["serving_default"]

# ---------------- ROUTES ---------------- #

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/result')
def result():
    return render_template('404.html')   # ✅ create this file

# ---------------- PREDICTION ---------------- #

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0   # 🔥 float32 FIX
    img = np.expand_dims(img, axis=0)
    return img


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"status": "error"})

    f = request.files['file']
    filename = secure_filename(f.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    f.save(filepath)

    img = preprocess_image(filepath)

    # 🔥 KEY FIX: keyword argument + float32 tensor
    outputs = infer(keras_tensor=tf.constant(img, dtype=tf.float32))

    prediction = list(outputs.values())[0].numpy()[0][0]

    # Since: car = 0, non_car = 1
    car_confidence = 1 - prediction

    THRESHOLD = 0.7  # stricter threshold

    if car_confidence < THRESHOLD:
        return jsonify({
            "status": "no_car",
            "confidence": float(car_confidence)
        })

    return jsonify({
        "status": "car_detected",
        "confidence": float(car_confidence)
    })




# ---------------- MAIN ---------------- #

if __name__ == '__main__':
    app.run(debug=True)
