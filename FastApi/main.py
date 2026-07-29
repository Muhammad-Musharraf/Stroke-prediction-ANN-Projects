from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal, Annotated
import tensorflow as tf
import joblib                   
import os
import sentry_sdk
import dotenv
from datetime import datetime
import pandas as pd
import uvicorn
import logging

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Model & Pipeline ──────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR          = os.path.dirname(BASE_DIR)
MODEL_PATH        = os.path.join(ROOT_DIR, "Model", "stroke_prediction.keras")
TRANSFORMER_PATH  = os.path.join(ROOT_DIR, "Model", "column_trans.pkl")   

model        = tf.keras.models.load_model(MODEL_PATH)
column_trans = joblib.load(TRANSFORMER_PATH)                               

print("✅ Model Loaded")
print("✅ column_trans (OHE + StandardScaler) Loaded")

# ── Sentry ────────────────────────────────────────────────────────────────────
SENTRY_DSN = os.getenv("SENTRY_DSN")
sentry_sdk.init(dsn=SENTRY_DSN, send_default_pii=True)

# ── Label Map ─────────────────────────────────────────────────────────────────
STROKE_LABEL = {0: "No Stroke", 1: "Stroke"}

# ── Request Schema ────────────────────────────────────────────────────────────
class UserInput(BaseModel):
    gender:            Literal["Male", "Female"]               = Field(..., description="Gender")
    age:               Annotated[int,   Field(gt=0)]           = Field(..., description="Age in years")
    hypertension:      Literal[0, 1]                           = Field(..., description="0=No, 1=Yes")
    heart_disease:     Literal[0, 1]                           = Field(..., description="0=No, 1=Yes")
    Married:           Literal["Yes", "No"]                    = Field(..., description="Marital status")
    Work_Type:         Literal["Private", "Self-employed",
                               "Govt_job", "children",
                               "Never_worked"]                 = Field(..., description="Work type")
    Residence_type:    Literal["Urban", "Rural"]               = Field(..., description="Residence type")
    Avg_Glucose_Level: Annotated[float, Field(gt=0)]           = Field(..., description="Avg glucose mg/dL")
    bmi:               Annotated[float, Field(gt=0)]           = Field(..., description="BMI kg/m²")
    Smoking_Status:    Literal["formerly smoked", "never smoked",
                               "smokes", "Unknown"]            = Field(..., description="Smoking status")

# ── Response Schema ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    label:              str   = Field(..., description="'Stroke' or 'No Stroke'")
    stroke_probability: float = Field(..., description="Sigmoid output 0–1")
    timestamp:          str

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "Welcome to the Stroke Prediction API!"}

@app.get("/health")
def health():
    return {"status": "OK", "timestamp": datetime.now().isoformat()}

@app.get("/sentry-debug")
def trigger_error():
    division_by_zero = 1 / 0

# ── Prediction Endpoint ───────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict(user: UserInput):
    try:
        # ✅ FIX 1 — Column names EXACTLY match what notebook used for training
        #            (spaces not underscores — this is what column_trans was fitted on)
        input_df = pd.DataFrame([{
            "gender":            user.gender,
            "age":               user.age,
            "hypertension":      user.hypertension,
            "heart disease":     user.heart_disease,    
            "Married":           user.Married,           
            "Work Type":         user.Work_Type,         
            "Residence_type":    user.Residence_type,
            "Avg_Glucose_Level": user.Avg_Glucose_Level, 
            "bmi":               user.bmi,
            "Smoking Status":    user.Smoking_Status,    
        }])

        # FIX 2 — Use the saved pipeline instead of pd.get_dummies()
        #            column_trans applies BOTH OneHotEncoder + StandardScaler
        #            exactly as the model was trained — raw numbers are now scaled
        features = column_trans.transform(input_df)     # shape → (1, 20)

        # ── Predict ──────────────────────────────────────────────────────────
        raw_output  = model.predict(features, verbose=0)   # shape → (1, 1)
        stroke_prob = float(raw_output[0][0])
        label_idx   = int(stroke_prob >= 0.5)

        log.info("Prediction: %s (p=%.4f)", STROKE_LABEL[label_idx], stroke_prob)

        return PredictionResponse(
            label              = STROKE_LABEL[label_idx],
            stroke_probability = round(stroke_prob, 6),
            timestamp          = datetime.now().isoformat(),
        )
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))
    
# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)