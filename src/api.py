import os
import sys
import tempfile
import pandas as pd
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import logging
from google.cloud import storage

import ipdb


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings.setting import SERVICE_ACCOUNT_KEY_PATH, BUCKET_NAME, load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Supply Chain Demand Forecasting",
    version="1.0.0",

)


config = load_config()
DATA_CONFIG = config["data"]

model = None

features = [
    'Inventory Level', 'Units Sold', 'Units Ordered',
    'Price', 'Weather Condition', 'Holiday/Promotion', 'Seasonality',
    'Year', 'Month', 'Day', 'quarter', 'day_of_week', 'days_since_start',
    'demand_lag_7', 'demand_rolling_mean_30', 'discount_rate',
    'sell_through_rate', 'supply_gap'
]


def load_model_from_gcs(model_name="xgboost_model.json", ):
    try:
        storage_client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_KEY_PATH)
        bucket = storage_client.bucket(bucket_name=BUCKET_NAME)
        blob = bucket.blob(blob_name=f"models/{model_name}")
        local_path = os.path.join(tempfile.gettempdir(), model_name)
        blob.download_to_filename(local_path)
        model = xgb.XGBRegressor()
        model.load_model(local_path)
        logger.info(f"Model loaded successfully from GCS: {model_name}")
        return model
    except Exception as e:
        logger.info(f"Failed to load the model: {e}")

    

# try:
model = load_model_from_gcs()
# except Exception as e:
#     logger.warning(f"Model is not available to start: {str(e)}")


# Pydantic Models

class PredictionFreatures(BaseModel):
    inventory_level: float
    units_sold: float
    units_ordered: float
    price: float
    weather_condition: int
    holiday_promotion: int
    seasonality: int
    year: int
    month: int
    day: int
    quarter: int
    day_of_week: int
    days_since_start: int
    demand_lag_7: float 
    demand_rolling_mean_30: float
    discount_rate: float
    sell_through_rate: float
    supply_gap: float

    class Config:
        json_schema_extra = {
            "example": {
                "inventory_level": 410,
                "units_sold": 200,
                "units_ordered": 152,
                "price": 70.88,
                "weather_condition": 3,
                "holiday_promotion": 1,
                "seasonality": 2,
                "year": 2022,
                "month": 1,
                "day": 31,
                "quarter": 1,
                "day_of_week": 0,
                "days_since_start": 30,
                "demand_lag_7": 17.87,
                "demand_rolling_mean_30": 119.56,
                "discount_rate": 0.0,
                "sell_through_rate": 0.48,
                "supply_gap": 48

            }
        }


class PredictionRequest(BaseModel):
    features: PredictionFreatures



class PredictionResponse(BaseModel):
    predicted_demand: float
    confidence: str
    recommendation: str



@app.get("/", tags=["Info"])
def root():
    return {
        "Name": "Supply chain demand forecasting API....",
        "Version": "1.0.0",
        
    }



@app.get("/features", tags=["Info"])
def get_featutes():
    feature_info = {
        "total_features": len(features),
        "features": features,
        "description": "Features required to predict"
    }
    return feature_info



@app.post("/predict", tags=["Prediction"], response_model=PredictionResponse)
def predict_demand(request: PredictionRequest):

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available")
    
    try:
        features_dict = request.features.model_dump(exclude_none=True)
        name_mapping = {
            'inventory_level':'Inventory Level', 
            'units_sold':'Units Sold', 
            'units_ordered':'Units Ordered', 
            'price':'Price', 
            'weather_condition':'Weather Condition', 
            'holiday_promotion':'Holiday/Promotion', 
            'seasonality':'Seasonality',
            'year':'Year', 
            'month':'Month', 
            'day':'Day', 
        }
        
    
        for old_name, new_name in name_mapping.items():
            if old_name in features_dict:
                features_dict[new_name] = features_dict.pop(old_name)

        
        input_df = pd.DataFrame([features_dict])
        input_df = input_df[features]

        prediction = model.predict(input_df)[0]
        # ipdb.set_trace()


        if prediction > 200:
            confidence = "High"
            recommendation = "Increase stock levels - high demand expected"
        elif prediction > 100:
            confidence = "Medium"
            recommendation = "Maintain current stock levels"
        else:
            confidence = "low"
            recommendation = "Consider reducing the stock - lower demand expected"
        
        logger.info(f"Prediction made: {prediction: .2f} Confidence: {confidence}")

        return {
            "predicted_demand": float(prediction),
            "confidence": confidence,
            "recommendation": recommendation

        }
    
    except Exception as e:
        logger.info(f"Error occured while predicting: {e}")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app, host="localhost", port=8080)




    
    
        














