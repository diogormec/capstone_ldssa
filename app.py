# Import necessary libraries
import os
import json
import joblib
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from prepare_features import preparar_features_para_predicao

# Database dependencies
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError, CharField, DateTimeField
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
import datetime

# ============================
# Database Configuration
# ============================
DB = SqliteDatabase('predictions.db')

class PredictionPrice(Model):
    observation_id = CharField(unique=True)
    sku = IntegerField()
    date = CharField()
    competitor = CharField()
    observation = TextField()
    predicted_price = FloatField()
    actual_price = FloatField(null=True)
    created_at = CharField(default=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    class Meta:
        database = DB

DB.create_tables([PredictionPrice], safe=True)
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

# ============================
# Load Required Artifacts
# ============================

loaded_models = {}
loaded_features_data = {}

# Log local directory and ensure 'models' folder exists
print("cwd:", os.getcwd())
assert os.path.exists("models"), "'models' folder not found!"

def load_all_competitor_data():
    print("Starting model loading...")

    competitors = ['competitorA', 'competitorB']

    for comp in competitors:
        try:
            print(f"Loading model for {comp}...")
            model = joblib.load(f"models/model_{comp}.pkl")
            features_data = joblib.load(f"models/features_data_{comp}.pkl")
            loaded_models[comp] = model
            loaded_features_data[comp] = features_data
            print(f"Successfully loaded model and data for {comp}")

        except Exception as e:
            print(f"Erro ao carregar dados para {comp}: {str(e)}")

    print("Closing models loading task.")

# Chamada da função de carregamento
load_all_competitor_data()

# Confirmar modelos carregados
print("Models loaded:", list(loaded_models.keys()))

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
    loaded = list(loaded_models.keys())
    return jsonify({
        "message": "Price prediction API is running!",
        "loaded_models": loaded
    })

# =================
# Prediction Endpoint
# =================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        obs = data.get("data")
        observation_id = data.get("observation_id")

        if not observation_id:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            import random
            import string
            random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
            observation_id = f"pred_{timestamp}_{random_suffix}"

        if not obs:
            return jsonify({"error": "Missing 'data' object with input details"}), 400

        existing_prediction = None
        try:
            existing_prediction = PredictionPrice.get(PredictionPrice.observation_id == observation_id)
        except PredictionPrice.DoesNotExist:
            pass

        if existing_prediction:
            if data.get("observation_id"):
                return jsonify({
                    "observation_id": observation_id,
                    "error": f"ERROR: Observation ID '{observation_id}' already exists."
                }), 400
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
                observation_id = f"pred_{timestamp}_{random_suffix}"

        sku = int(obs.get("sku"))
        date_str = obs.get("date")
        competitor = obs.get("competitor")

        if not all([sku, date_str, competitor]):
            return jsonify({
                "observation_id": observation_id,
                "error": "'sku', 'date', and 'competitor' are required fields."
            }), 400

        if competitor not in loaded_models:
            return jsonify({
                "observation_id": observation_id,
                "error": f"Competitor '{competitor}' not found in loaded models."
            }), 400

        features_data = loaded_features_data[competitor]
        columns = features_data["columns"]
        features = features_data["features"]
        model = loaded_models[competitor]
        date = pd.to_datetime(date_str)

        predicted_price = preparar_features_para_predicao(
            sku, date, competitor,
            sales_df_clean, prices_df_clean, campaigns_df_clean,
            features, model
        )

        try:
            prediction = PredictionPrice(
                observation_id=observation_id,
                sku=sku,
                date=date.strftime("%Y-%m-%d"),
                competitor=competitor,
                observation=json.dumps(obs),
                predicted_price=float(predicted_price))
            prediction.save()
        except IntegrityError:
            DB.rollback()
            return jsonify({
                "observation_id": observation_id,
                "error": f"ERROR: Observation ID '{observation_id}' already exists"
            }), 400

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

@app.route("/update_actual/<observation_id>", methods=["POST"])
def update_actual(observation_id):
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

@app.route("/predictions", methods=["GET"])
def get_predictions():
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
    port = os.environ.get('PORT')
    app.run(host="0.0.0.0", port=port, debug=True)
