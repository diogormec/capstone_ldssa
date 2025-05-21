# 📊 Competitor Price Forecasting API - Retailz Project

## 🚀 Overview

This project provides a REST API for predicting competitor prices (Competitor A and B) to help Retailz optimize their pricing strategies. The API supports:

- 🔮 Price predictions for future dates
- 📝 Logging of actual prices for model improvement
- 📊 Retrieval of historical predictions

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
   source .venv/bin/activate # On Mac/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Running the API

```bash
python app.py
```

The API will be available at `http://localhost:5000`

## 📚 API Endpoints

### 🔮 Make a Prediction
**POST** `/predict`

```powershell
# Example 1: Automatic observation ID
Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method Post -Body '{"data": {"sku": "4443", "date": "2025-05-20", "competitor": "competitorA"}}' -ContentType "application/json"

# Example 2: Custom observation ID
Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method Post -Body '{"observation_id": "my_custom_id", "data": {"sku": "4443", "date": "2025-05-20", "competitor": "competitorA"}}' -ContentType "application/json"
```

### 🔄 Update Actual Price
**POST** `/update_actual/<observation_id>`

```powershell
Invoke-RestMethod -Uri "http://localhost:5000/update_actual/my_custom_id" -Method Post -Body '{"actual_price": 125.50}' -ContentType "application/json"
```

### 📜 Get All Predictions
**GET** `/predictions`

```powershell
Invoke-RestMethod -Uri "http://localhost:5000/predictions" -Method Get
```


## 📊 Key Insights from Analysis

################################

## 🤖 Model Details

- **Algorithm**: Gradient Boosting (LightGBM/XGBoost)
- **Features**:
  - Historical prices and discounts
  - ~~Competitor price lags~~
  - Calendar features (day-of-week, month)
  - Promotional indicators
- **Evaluation**: Category-weighted MAE with <10% variance constraint


---

Developed by Diogo Ramalho for Retailz as part of the Lisbon Data Science Academy program
