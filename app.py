# app.py

import os
import json
import joblib
import pandas as pd
import datetime
from flask import Flask, request, jsonify

# Import custom feature preparation function
from prepare_features import prepare_features_for_prediction

# Database dependencies
from peewee import (
    Model, IntegerField, FloatField, CharField, IntegrityError
)
from playhouse.db_url import connect
from playhouse.shortcuts import model_to_dict

# ============================
# Database Configuration
# ============================

# Connect to PostgreSQL via DATABASE_URL or fallback to local SQLite for development
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

# Updated PredictionPrice model to store both competitors' prices in one row
class PredictionPrice(Model):
    sku = CharField()
    time_key = IntegerField()
    pvp_is_competitorA = FloatField(null=True)
    pvp_is_competitorB = FloatField(null=True)
    pvp_is_competitorA_actual = FloatField(null=True)
    pvp_is_competitorB_actual = FloatField(null=True)

    class Meta:
        database = DB
        indexes = (
            # Unique constraint on SKU and time_key (one row per SKU/time_key)
            (('sku', 'time_key'), True),
        )

# Create the table if it doesn't exist
DB.create_tables([PredictionPrice], safe=True)

# ============================
# Load ML Models & Features
# ============================

loaded_models = {}
loaded_features_data = {}

assert os.path.exists("models"), "'models' directory not found!"

def load_all_competitor_data():
    print("Loading models and features for competitors...")
    competitors = ['competitorA', 'competitorB']
    for comp in competitors:
        try:
            loaded_models[comp] = joblib.load(f"models/model_{comp}.pkl")
            loaded_features_data[comp] = joblib.load(f"models/features_data_{comp}.pkl")
            print(f"Loaded model and features for {comp}")
        except Exception as e:
            print(f"Failed to load model/features for {comp}: {e}")

load_all_competitor_data()

# Load auxiliary dataframes required for feature preparation
sales_df_clean = pd.read_parquet("sales_df_clean.parquet")
prices_df_clean = pd.read_parquet("prices_df_clean.parquet")
campaigns_df_clean = pd.read_parquet("campaigns_df_clean.parquet")

# ============================
# Initialize Flask App
# ============================

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    """Simple health check endpoint."""
    return jsonify({"message": "Forecast price API is live."})

# ============================
# Forecast Prices Endpoint
# ============================

@app.route("/forecast_prices/", methods=["POST"])
def forecast_prices():
    """
    Accepts JSON with 'sku' (string) and 'time_key' (int, YYYYMMDD format),
    predicts prices for both competitors and saves to DB.
    Returns predicted prices in JSON.
    """
    try:
        data = request.get_json()
        sku = data.get("sku")
        time_key = data.get("time_key")

        # Validate input
        if not sku or not isinstance(time_key, int):
            return jsonify({
                "error": "Invalid input format. Required fields: 'sku' (string), 'time_key' (integer YYYYMMDD)"
            }), 422

        # Prepare to store predictions per competitor
        predictions = {}

        # Predict for each competitor and store results
        for competitor in ['competitorA', 'competitorB']:
            if competitor not in loaded_models:
                return jsonify({"error": f"Model for {competitor} not loaded."}), 500

            model = loaded_models[competitor]
            features_data = loaded_features_data[competitor]
            features = features_data["features"]

            # Convert time_key integer YYYYMMDD to datetime object
            date = pd.to_datetime(str(time_key), format="%Y%m%d")

            # Prepare features and predict price
            predicted_price = prepare_features_for_prediction(
                int(sku), date, competitor,
                sales_df_clean, prices_df_clean, campaigns_df_clean,
                features, model
            )

            predictions[f"pvp_is_{competitor}"] = float(predicted_price)

        # Save or update prediction record in DB
        try:
            # Try to create new record
            PredictionPrice.create(
                sku=sku,
                time_key=time_key,
                pvp_is_competitorA=predictions.get("pvp_is_competitorA"),
                pvp_is_competitorB=predictions.get("pvp_is_competitorB")
            )
        except IntegrityError:
            # If record already exists (unique constraint), optionally update here or skip
            DB.rollback()  # Rollback current transaction
            # Optional: Implement update logic if you want to overwrite existing predictions

        # Prepare response JSON
        response = {"sku": sku, "time_key": time_key}
        response.update(predictions)

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "An error occurred during prediction."
        }), 500

# ============================
# Actual Prices Update Endpoint
# ============================

@app.route("/actual_prices/", methods=["POST"])
def actual_prices():
    """
    Accepts JSON with 'sku', 'time_key', and optionally
    'pvp_is_competitorA_actual' and/or 'pvp_is_competitorB_actual'.
    Updates the actual prices in DB and returns updated record.
    """
    try:
        data = request.get_json()
        sku = data.get("sku")
        time_key = data.get("time_key")
        pvp_a_actual = data.get("pvp_is_competitorA_actual")
        pvp_b_actual = data.get("pvp_is_competitorB_actual")

        # Validate input
        if not sku or not isinstance(time_key, int):
            return jsonify({
                "error": "Invalid input format. Required fields: 'sku' (string), 'time_key' (integer YYYYMMDD)"
            }), 422

        # Retrieve existing prediction record
        record = PredictionPrice.get_or_none(
            (PredictionPrice.sku == sku) &
            (PredictionPrice.time_key == time_key)
        )

        if record is None:
            return jsonify({
                "error": f"No prediction record found for sku '{sku}' and time_key '{time_key}'."
            }), 404

        # Update actual prices if provided
        if pvp_a_actual is not None:
            record.pvp_is_competitorA_actual = float(pvp_a_actual)
        if pvp_b_actual is not None:
            record.pvp_is_competitorB_actual = float(pvp_b_actual)

        record.save()

        # Return updated record as JSON
        return jsonify(model_to_dict(record))

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "An error occurred while updating actual prices."
        }), 500

# ============================
# Retrieve All Prediction Records Endpoint
# ============================

@app.route("/records/", methods=["GET"])
def get_all_records():
    """
    Returns all prediction records sorted by newest first.
    """
    try:
        records = [
            model_to_dict(record) for record in
            PredictionPrice.select().order_by(PredictionPrice.time_key.desc())
        ]
        return jsonify(records)
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "An error occurred while fetching records."
        }), 500

# ============================
# Run Flask App
# ============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
