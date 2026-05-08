from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle

app = FastAPI(title="FMCG Promo Forecast API")

# Load model and features
with open('models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

# Load historical data for lag/rolling lookups
df = pd.read_csv('data/processed/train_features.csv')

class PredictRequest(BaseModel):
    store_id: int
    day_of_week: int
    promo: int
    state_holiday: int
    school_holiday: int
    store_type: int
    assortment: int
    competition_distance: float
    promo2: int
    year: int
    month: int
    day: int
    week_of_year: int
    day_of_year: int

def get_store_features(store_id: int):
    """Get latest lag and rolling features for a store."""
    store_data = df[df['Store'] == store_id].sort_values('Date')
    if len(store_data) == 0:
        return {}
    latest = store_data.iloc[-1]
    return {
        'Sales_Lag_7': latest.get('Sales_Lag_7', 0),
        'Sales_Lag_14': latest.get('Sales_Lag_14', 0),
        'Sales_Lag_28': latest.get('Sales_Lag_28', 0),
        'Sales_Rolling_Mean_7': latest.get('Sales_Rolling_Mean_7', 0),
        'Sales_Rolling_Mean_14': latest.get('Sales_Rolling_Mean_14', 0),
        'Sales_Rolling_Mean_28': latest.get('Sales_Rolling_Mean_28', 0),
        'Sales_Rolling_Std_7': latest.get('Sales_Rolling_Std_7', 0),
        'Sales_Rolling_Std_14': latest.get('Sales_Rolling_Std_14', 0),
        'Sales_Rolling_Std_28': latest.get('Sales_Rolling_Std_28', 0),
        'PromoRunLength': latest.get('PromoRunLength', 0),
        'PromoYesterday': latest.get('PromoYesterday', 0),
        'CompetitionOpenMonths': latest.get('CompetitionOpenMonths', 0),
    }

@app.get("/")
def root():
    return {"message": "FMCG Promo Forecast API is running"}

@app.post("/predict-sales")
def predict_sales(request: PredictRequest):
    store_feats = get_store_features(request.store_id)
    
    input_data = {
        'Store': request.store_id,
        'DayOfWeek': request.day_of_week,
        'Promo': request.promo,
        'StateHoliday': request.state_holiday,
        'SchoolHoliday': request.school_holiday,
        'StoreType': request.store_type,
        'Assortment': request.assortment,
        'CompetitionDistance': request.competition_distance,
        'Promo2': request.promo2,
        'Year': request.year,
        'Month': request.month,
        'Day': request.day,
        'WeekOfYear': request.week_of_year,
        'DayOfYear': request.day_of_year,
        'IsWeekend': 1 if request.day_of_week in [6, 7] else 0,
        'IsMonthStart': 1 if request.day <= 3 else 0,
        'IsMonthEnd': 1 if request.day >= 28 else 0,
        'Quarter': (request.month - 1) // 3 + 1,
        **store_feats
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)
    
    prediction = model.predict(input_df)[0]
    prediction = max(0, float(prediction))

    return {
        "store_id": request.store_id,
        "predicted_sales": round(prediction, 2),
        "promo_active": bool(request.promo)
    }

@app.post("/predict-uplift")
def predict_uplift(request: PredictRequest):
    # Predict without promo
    request.promo = 0
    no_promo = predict_sales(request)
    
    # Predict with promo
    request.promo = 1
    with_promo = predict_sales(request)

    uplift = with_promo['predicted_sales'] - no_promo['predicted_sales']
    uplift_pct = (uplift / no_promo['predicted_sales']) * 100

    return {
        "store_id": request.store_id,
        "sales_without_promo": no_promo['predicted_sales'],
        "sales_with_promo": with_promo['predicted_sales'],
        "uplift_value": round(uplift, 2),
        "uplift_percentage": round(uplift_pct, 2)
    }

@app.post("/simulate-promotion")
def simulate_promotion(request: PredictRequest):
    uplift_data = predict_uplift(request)
    
    cost_per_day = 500
    roi = ((uplift_data['uplift_value'] - cost_per_day) 
           / cost_per_day) * 100

    recommendation = "RUN PROMOTION" if roi > 0 else "DO NOT RUN"

    return {
        **uplift_data,
        "estimated_promo_cost": cost_per_day,
        "estimated_roi_percentage": round(roi, 2),
        "recommendation": recommendation
    }