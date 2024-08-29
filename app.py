from flask import Flask, request, render_template, redirect
import os
import tensorflow as tf
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Ensure the upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load your pre-trained model at startup
try:
    # model = tf.keras.models.load_model('nutrient.h5')
    model = tf.keras.models.load_model('nutrient.keras')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Adjust size as needed
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        processed_image = preprocess_image(filepath)

        if model:
            try:
                prediction = model.predict(processed_image)
                result = np.argmax(prediction, axis=1)
                # Assuming you have a mapping of labels
                labels = ['Zinc Deficiency', 'Potassium Deficiency', 'Magnesium Deficiency', 'Phosphorus Deficiency']
                predicted_label = labels[result[0]]  # Map the result to the label
                return render_template('index.html', prediction_text=f'Prediction: {predicted_label}', image_path=filepath)
            except Exception as e:
                return f"Prediction error: {e}"
        else:
            return "Model is not loaded properly."

if __name__ == '__main__':
    app.run(port=8000)
