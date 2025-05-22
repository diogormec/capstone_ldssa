# 📊 Competitor Price Forecasting API - Retailz Project

## 🚀 Overview

This project provides a RESTful API that enables:

* 🔮 Forecasting prices for two competitors (`competitorA` and `competitorB`) based on a given SKU and date.
* 📝 Logging and updating actual competitor prices after the forecast date.
* 📂 Retrieving all prediction records from the database.

The API is designed to support pricing intelligence and strategy for Retailz by enabling fast access to model forecasts and post-hoc performance tracking.

## 🔧 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/retailz-price-forecasting.git
   cd retailz-price-forecasting
   ```

2. **Set up a virtual environment**

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # On Windows
   source .venv/bin/activate # On macOS/Linux
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the following files and folders are available:**

   * `models/model_competitorA.pkl`
   * `models/model_competitorB.pkl`
   * `models/features_data_competitorA.pkl`
   * `models/features_data_competitorB.pkl`
   * `sales_df_clean.parquet`
   * `prices_df_clean.parquet`
   * `campaigns_df_clean.parquet`

## 🏃‍♂️ Running the API

```bash
python app.py
```

The API will be available at:
📍 `http://localhost:5000`

## 📚 API Endpoints

### 🔮 Forecast Prices

**POST** `/forecast_prices/`

Forecast competitor prices for a given `sku` and `time_key` (in `YYYYMMDD` format).

**Request Body (JSON):**

```json
{
  "sku": "4443",
  "time_key": 20250520
}
```

**Response Example:**

```json
{
  "sku": "4443",
  "time_key": 20250520,
  "pvp_is_competitorA": 124.87,
  "pvp_is_competitorB": 129.45
}
```

---

### 📝 Update Actual Prices

**POST** `/actual_prices/`

Update actual observed competitor prices for a specific `sku` and `time_key`.

**Request Body (JSON):**

```json
{
  "sku": "4443",
  "time_key": 20250520,
  "pvp_is_competitorA_actual": 126.00,
  "pvp_is_competitorB_actual": 131.50
}
```

**Response:** JSON object with the updated record.

---

### 📂 Retrieve All Prediction Records

**GET** `/records/`

Returns all prediction records in descending order by `time_key`.

---

## 🧠 Model Details

* **Models**: One LightGBM model per competitor.
* **Inputs**:

  * Cleaned sales, pricing, and promotional data.
  * Engineered features like calendar variables, recent sales patterns, and promotion flags.
* **Output**: Forecasted price for a given `sku` and date.
* **Storage**: All forecasts and updates are persisted in a PostgreSQL or SQLite database.

## ✅ Example Use Cases

* Predict competitor prices for a specific date and product.
* Compare model forecast with actual prices after the date.
* Track prediction performance over time.

---

Developed by **Diogo Ramalho** for **Retailz**
as part of the **Lisbon Data Science Academy (LDSA) Capstone Project**

---
