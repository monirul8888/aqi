from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from io import BytesIO

# Load model
model = tf.keras.models.load_model('best_aqi_model.keras')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']

        # Read image as BytesIO
        img = load_img(BytesIO(image_file.read()), target_size=(128, 128))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Metadata
        place = int(request.form.get('place', 0))
        time = float(request.form.get('time', 12.0))
        month = float(request.form.get('month', 1.0))

        place_arr = np.array([[place]])
        time_arr = np.array([[time / 24.0]])
        month_arr = np.array([[month / 12.0]])

        # Prediction
        pred = model.predict({
            'img_input': img,
            'place_input': place_arr,
            'time_input': time_arr,
            'month_input': month_arr
        })

        pm25 = float(pred[0][0])

        if 'submit' in request.form:
            return render_template('index.html', prediction=pm25)
        return jsonify({'PM2.5': pm25})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
