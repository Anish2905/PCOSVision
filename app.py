from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("final_model.h5")

# Define class names
class_names = ["Infected", "Not Infected"]  # Ensure correct order

def preprocess_image(image):
    """Resize, normalize, and prepare the image for prediction."""
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize pixel values
    return np.expand_dims(img, axis=0)  # Add batch dimension

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Process and predict
    img = preprocess_image(file)
    preds = model.predict(img)
    result = class_names[np.argmax(preds)]  # Get class label

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
