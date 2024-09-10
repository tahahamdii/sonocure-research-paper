import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import torch
import cv2
import torchvision.transforms as tt
from PIL import Image
import io
import base64
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
# Define the UNet model (same as your original code)
import os  # Add this import statement
from io import BytesIO  # Add this import statement
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)
CORS(app)
# traitement ultrason
OUTPUT_DIR = "../Sonocure/src/assets/OutputImages"
MODEL_PATH = 'model_traitement/my_model.pkl'
ENCODER_PATH = 'model_traitement/encoder.pkl'
SCALER_PATH = 'model_traitement/scaler.pkl'
SCALERY_PATH = 'model_traitement/scaler_y.pkl'
X_PREDICTION_PATH = 'model_traitement/X_prediction.csv'
ULTRASOUND_MODEL_PATH = 'model_detection/ultrasound_model_normalized.h5'
ULTRASOUND_INPUT_SCALER_PATH = 'model_detection/input_scaler.joblib'
ULTRASOUND_OUTPUT_SCALER_PATH = 'model_detection/output_scaler.joblib'
# Load models and scalers
treatment_model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)
scaler_y = joblib.load(SCALERY_PATH)
X_prediction = pd.read_csv(X_PREDICTION_PATH)

ultrasound_model = load_model(ULTRASOUND_MODEL_PATH)
input_scaler = joblib.load(ULTRASOUND_INPUT_SCALER_PATH)
output_scaler = joblib.load(ULTRASOUND_OUTPUT_SCALER_PATH)

# Features
numeric_features = ['Tumor Size (cm)', 'Age']
categorical_features = ['Tumor Location', 'Sex', 'Tumor Type']

# Fonction de prétraitement des données d'entrée
def preprocess_input(example_data, X_prediction):
    example_df = pd.DataFrame([example_data])
    X_combined = pd.concat([X_prediction, example_df], ignore_index=True)

    # Encoding
    encoded_features = encoder.transform(X_combined[categorical_features])
    encoded_columns = encoder.get_feature_names_out(categorical_features)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns)
    encoded_df = pd.concat([X_combined.drop(columns=categorical_features), encoded_df], axis=1)

    # Scaling
    encoded_df[numeric_features] = scaler.transform(encoded_df[numeric_features])

    # Extraction of the encoded and scaled data example
    example_processed = encoded_df.iloc[-1]
    example_processed_df = pd.DataFrame(example_processed).T
    return example_processed_df


# Fonction de prédiction
# Preprocessing function
def preprocess_input(example_data: dict, X_prediction: pd.DataFrame) -> pd.DataFrame:
    try:
        example_df = pd.DataFrame([example_data])
        X_combined = pd.concat([X_prediction, example_df], ignore_index=True)

        # Encode categorical features
        encoded_features = encoder.transform(X_combined[categorical_features])
        encoded_columns = encoder.get_feature_names_out(categorical_features)
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns)
        combined_df = pd.concat([X_combined.drop(columns=categorical_features), encoded_df], axis=1)

        # Scale numeric features
        combined_df[numeric_features] = scaler.transform(combined_df[numeric_features])

        # Return the processed last row (input data)
        return combined_df.iloc[[-1]]
    except Exception as e:
        raise ValueError(f"Error during preprocessing: {str(e)}")


# Prediction function for treatment
def predict_treatment(example_data: dict) -> np.ndarray:
    try:
        processed_data = preprocess_input(example_data, X_prediction)
        normalized_prediction = treatment_model.predict(processed_data)
        return scaler_y.inverse_transform(normalized_prediction)
    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}")
# Route pour la prédiction
@app.route('/predict_traitement', methods=['POST'])
def predict_traitement():
    print(f"Content-Type: {request.content_type}")  # Log content type

    if request.content_type != 'application/json':
        return jsonify({'error': 'Unsupported Media Type. Expected application/json'}), 415

    try:
        input_data = request.get_json()
        print(f"Received JSON: {input_data}")  # Log received JSON

        if input_data is None:
            return jsonify({'error': 'No JSON data provided'}), 400

        result = predict_treatment(input_data)
        return jsonify({'prediction': result.tolist()})
    except Exception as e:
        print(f"Error: {str(e)}")  # Log exception details
        return jsonify({'error': str(e)}), 500
# Predict ultrasound
ultrasound_model = load_model('model_detection/ultrasound_model_normalized.h5')
input_scaler = joblib.load('model_detection/input_scaler.joblib')
output_scaler = joblib.load('model_detection/output_scaler.joblib')

