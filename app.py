# Import necessary libraries
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
    Model, IntegerField, FloatField, TextField,
    CharField, IntegrityError
)
from playhouse.db_url import connect
from playhouse.shortcuts import model_to_dict

# ============================
# Database Configuration
# ============================

# Connect to PostgreSQL (Railway) or fallback to SQLite locally
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

# Define the database model for storing predictions
class PredictionPrice(Model):
    sku = CharField()
    time_key = IntegerField()
    competitor = CharField()
    predicted_price = FloatField()
    actual_price = FloatField(null=True)
    created_at = CharField(default=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    class Meta:
        database = DB
        indexes = (
            # Unique constraint on (sku, time_key, competitor)
            (('sku', 'time_key', 'competitor'), True),
        )

# Create table if it doesn't already exist
DB.create_tables([PredictionPrice], safe=True)

# ============================
# Load Required Artifacts
# ============================

# Dictionaries to store models and corresponding features
loaded_models = {}
loaded_features_data = {}

# Ensure the models directory exists
assert os.path.exists("models"), "'models' folder not found!"

# Load models and features for all competitors
def load_all_competitor_data():
    print("Loading models...")
    competitors = ['competitorA', 'competitorB']

    for comp in competitors:
        try:
            model = joblib.load(f"models/model_{comp}.pkl")
            features_data = joblib.load(f"models/features_data_{comp}.pkl")
            loaded_models[comp] = model
            loaded_features_data[comp] = features_data
            print(f"Loaded model and features for {comp}")
        except Exception as e:
            print(f"Failed to load model for {comp}: {e}")

# Load on startup
load_all_competitor_data()

# Load auxiliary dataframes
sales_df_clean = pd.read_parquet("sales_df_clean.parquet")
prices_df_clean = pd.read_parquet("prices_df_clean.parquet")
campaigns_df_clean = pd.read_parquet("campaigns_df_clean.parquet")

# ============================
# Initialize Flask App
# ============================

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Forecast price API is live."})

# ============================
# Forecast Prices Endpoint
# ============================

@app.route("/forecast_prices/", methods=["POST"])
def forecast_prices():
    try:
        # Parse incoming JSON
        data = request.get_json()

        # Validate input
        sku = data.get("sku")
        time_key = data.get("time_key")

        if not sku or not isinstance(time_key, int):
            return jsonify({"error": "Invalid input format. Required: 'sku' (string), 'time_key' (integer)"}), 422

        results = {"sku": sku, "time_key": time_key}

        # Predict price for each competitor
        for competitor in ['competitorA', 'competitorB']:
            if competitor not in loaded_models:
                return jsonify({"error": f"Model for {competitor} not loaded."}), 500

            model = loaded_models[competitor]
            features_data = loaded_features_data[competitor]
            features = features_data["features"]

            # Convert time_key to datetime
            date = pd.to_datetime(str(time_key), format="%Y%m%d")

            # Prepare and predict
            predicted_price = prepare_features_for_prediction(
                int(sku), date, competitor,
                sales_df_clean, prices_df_clean, campaigns_df_clean,
                features, model
            )

            # Store prediction in DB
            try:
                PredictionPrice.create(
                    sku=sku,
                    time_key=time_key,
                    competitor=competitor,
                    predicted_price=predicted_price
                )
            except IntegrityError:
                DB.rollback()  # Skip if already exists

            # Add prediction to response
            results[f"pvp_is_{competitor}"] = float(predicted_price)

        return jsonify(results)

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
    try:
        data = request.get_json()
        sku = data.get("sku")
        time_key = data.get("time_key")
        pvp_a_actual = data.get("pvp_is_competitorA_actual")
        pvp_b_actual = data.get("pvp_is_competitorB_actual")

        if not sku or not isinstance(time_key, int):
            return jsonify({"error": "Invalid input format. Required: 'sku' (string), 'time_key' (integer)"}), 422

        results = {"sku": sku, "time_key": time_key}

        # Update actual prices for both competitors
        for competitor, actual_price in zip(
            ['competitorA', 'competitorB'],
            [pvp_a_actual, pvp_b_actual]
        ):
            if actual_price is not None:
                try:
                    pred = PredictionPrice.get(
                        (PredictionPrice.sku == sku) &
                        (PredictionPrice.time_key == time_key) &
                        (PredictionPrice.competitor == competitor)
                    )
                    pred.actual_price = float(actual_price)
                    pred.save()

                    results[f"pvp_is_{competitor}"] = float(pred.predicted_price)
                    results[f"pvp_is_{competitor}_actual"] = float(pred.actual_price)

                except PredictionPrice.DoesNotExist:
                    return jsonify({
                        "error": f"No prediction found for SKU '{sku}' and time_key '{time_key}' for {competitor}"
                    }), 422

        return jsonify(results)

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "An error occurred while updating actual prices."
        }), 500

# ============================
# Records Retrieval Endpoint
# ============================

@app.route("/records/", methods=["GET"])
def get_all_records():
    try:
        records = [
            model_to_dict(p) for p in PredictionPrice.select().order_by(PredictionPrice.created_at.desc())
        ]
        return jsonify(records)
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "An error occurred while fetching records."
        }), 500

# ============================
# Run the Server
# ============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
