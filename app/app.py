from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load('model/aqi_model_xgb.pkl')

class AQIInput(BaseModel):
    PM2_5: float
    PM10: float
    NO: float
    NO2: float
    NOx: float
    NH3: float
    CO: float
    SO2: float
    O3: float
    Benzene: float
    Toluene: float
    Xylene: float

@app.post("/predict")
def predict_aqi(data: AQIInput):
    input_array = np.array([[data.PM2_5, data.PM10, data.NO, data.NO2, data.NOx,
                             data.NH3, data.CO, data.SO2, data.O3,
                             data.Benzene, data.Toluene, data.Xylene]])
    prediction = model.predict(input_array)[0]
    return {"predicted_AQI": round(prediction, 2)}
