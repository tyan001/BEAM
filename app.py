from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'models/random_forest_8features.pkl'

with open(MODEL_PATH, 'rb') as f:
    model_info = pickle.load(f)

model = model_info['model']
features = model_info['features']
diagnosis_order = model_info['diagnosis_order']

@app.route('/')
def home():
    return render_template('index.html', features=features, diagnosis_order=diagnosis_order)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        data = request.json

        # Create input array in the correct feature order
        input_data = []
        for feature in features:
            value = float(data[feature])
            input_data.append(value)

        # Convert to numpy array and reshape for prediction
        input_array = np.array(input_data).reshape(1, -1)

        # Get prediction probabilities
        probabilities = model.predict_proba(input_array)[0]

        # Get predicted class
        predicted_class = model.predict(input_array)[0]
        predicted_label = diagnosis_order[predicted_class]

        # Create response with probabilities for each class
        result = {
            'predicted_class': predicted_label,
            'predicted_class_code': int(predicted_class),
            'probabilities': [
                {
                    'label': diagnosis_order[i],
                    'probability': float(probabilities[i]),
                    'percentage': float(probabilities[i] * 100)
                }
                for i in range(len(diagnosis_order))
            ]
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
