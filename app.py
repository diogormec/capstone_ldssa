'''
# Example 1: Sending a prediction request without providing an observation_id (it will be generated automatically)
Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method Post -Body '{"data": {"sku": "4443", "date": "2025-05-20", "competitor": "competitorA"}}' -ContentType "application/json"

# Example 2: Sending a prediction request with a custom observation_id
# Note: If the ID already exists, an error will be returned
Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method Post -Body '{"observation_id": "my_custom_id", "data": {"sku": "4443", "date": "2025-05-20", "competitor": "competitorA"}}' -ContentType "application/json"

# Example 3: Retrieving all stored predictions
Invoke-RestMethod -Uri "http://localhost:5000/predictions" -Method Get

# Example 4: Updating the actual price after a prediction is made
Invoke-RestMethod -Uri "http://localhost:5000/update_actual/my_custom_id" -Method Post -Body '{"actual_price": 125.50}' -ContentType "application/json"

'''


# Import necessary libraries
import os
import json
import joblib
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from prepare_features import preparar_features_para_predicao  # Custom function for feature processing

# Database dependencies
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError, CharField, DateTimeField
)
from playhouse.shortcuts import model_to_dict
import datetime

# ============================
# Database Configuration
# ============================
# Using SQLite for local storage
DB = SqliteDatabase('predictions.db')

# Definition of the PredictionPrice table
class PredictionPrice(Model):
    observation_id = CharField(unique=True)
    sku = IntegerField()
    date = CharField()  # Stored as string (YYYY-MM-DD)
    competitor = CharField()
    observation = TextField()  # Stores input data as JSON string
    predicted_price = FloatField()
    actual_price = FloatField(null=True)  # Can be updated later
    created_at = CharField(default=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    class Meta:
        database = DB

# Create the table if it doesn't exist
DB.create_tables([PredictionPrice], safe=True)

# ============================
# Load Required Artifacts
# ============================

# Load registry of models per competitor (includes model, columns, dtypes, features)
models_registry = joblib.load("models_registry.pkl")

# Load cleaned datasets
sales_df_clean = pd.read_parquet("sales_df_clean.parquet")
prices_df_clean = pd.read_parquet("prices_df_clean.parquet")
campaigns_df_clean = pd.read_parquet("campaigns_df_clean.parquet")

# =================
# Flask API Initialization
# =================
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Price prediction API is running!"})

# =================
# Prediction Endpoint
# =================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        obs = data.get("data")  # Extract the input data dictionary
        
        # Allow optional observation_id
        observation_id = data.get("observation_id")
        
        # If no ID provided, generate one automatically
        if not observation_id:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            import random
            import string
            random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
            observation_id = f"pred_{timestamp}_{random_suffix}"

        # Validate input presence
        if not obs:
            return jsonify({"error": "Missing 'data' object with input details"}), 400

        # Check if observation_id already exists
        existing_prediction = None
        try:
            existing_prediction = PredictionPrice.get(PredictionPrice.observation_id == observation_id)
        except PredictionPrice.DoesNotExist:
            pass  # OK to proceed

        # If ID was provided explicitly and already exists, return error
        if existing_prediction:
            if data.get("observation_id"):
                return jsonify({
                    "observation_id": observation_id,
                    "error": f"ERROR: Observation ID '{observation_id}' already exists. Use a different one or omit to generate automatically."
                }), 400
            else:
                # Try generating a new one (very rare case)
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
                observation_id = f"pred_{timestamp}_{random_suffix}"

        # Extract individual inputs
        sku = int(obs.get("sku"))
        date_str = obs.get("date")
        competitor = obs.get("competitor")

        # Validate required fields
        if not all([sku, date_str, competitor]):
            return jsonify({
                "observation_id": observation_id,
                "error": "'sku', 'date', and 'competitor' are required fields."
            }), 400

        # Validate competitor existence in model registry
        if competitor not in models_registry:
            return jsonify({
                "observation_id": observation_id,
                "error": f"Competitor '{competitor}' not found in model registry."
            }), 400

        # Extract metadata
        columns = models_registry[competitor]["columns"]
        dtypes = models_registry[competitor]["dtypes"]
        features = models_registry[competitor]["features"]
        date = pd.to_datetime(date_str)
        model = models_registry[competitor]["model"]

        # Call feature preparation and prediction function
        predicted_price = preparar_features_para_predicao(
            sku, date, competitor,
            sales_df_clean, prices_df_clean, campaigns_df_clean,
            features, model
        )

        # Save prediction in the database
        try:
            prediction = PredictionPrice(
                observation_id=observation_id,
                sku=sku,
                date=date.strftime("%Y-%m-%d"),
                competitor=competitor,
                observation=json.dumps(obs),
                predicted_price=float(predicted_price)
            )
            prediction.save()
        except IntegrityError:
            DB.rollback()
            return jsonify({
                "observation_id": observation_id,
                "error": f"ERROR: Observation ID '{observation_id}' already exists"
            }), 400

        # Return prediction
        return jsonify({
            "observation_id": observation_id,
            "competitor": competitor,
            "prediction": float(predicted_price)
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "An error occurred while processing the prediction."
        }), 500

# =========================
# Update Actual Price Endpoint
# =========================
@app.route("/update_actual/<observation_id>", methods=["POST"])
def update_actual(observation_id):
    """Updates the actual price for a given prediction ID"""
    try:
        data = request.get_json()
        actual_price = data.get("actual_price")

        if actual_price is None:
            return jsonify({"error": "Missing 'actual_price' field"}), 400

        try:
            prediction = PredictionPrice.get(PredictionPrice.observation_id == observation_id)
            prediction.actual_price = float(actual_price)
            prediction.save()

            return jsonify({
                "observation_id": observation_id,
                "message": "Actual price successfully updated",
                "prediction": float(prediction.predicted_price),
                "actual_price": float(prediction.actual_price)
            })

        except PredictionPrice.DoesNotExist:
            return jsonify({
                "error": f"Prediction with ID '{observation_id}' not found"
            }), 404

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "An error occurred while updating the actual price."
        }), 500

# =========================
# Retrieve All Predictions
# =========================
@app.route("/predictions", methods=["GET"])
def get_predictions():
    """Returns all stored predictions"""
    try:
        predictions = [model_to_dict(p) for p in PredictionPrice.select()]
        return jsonify(predictions)

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "An error occurred while retrieving predictions."
        }), 500

# =========================
# Start the API Server
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to port 5000
    app.run(host="0.0.0.0", port=port, debug=True)  # Run the Flask app
