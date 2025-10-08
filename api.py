from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd, joblib, json

app = FastAPI()

# Load model + feature list
model = joblib.load("blinkit_discount_model.pkl")
feats = json.load(open("feature_list.json"))

class Record(BaseModel):
    data: dict

def recommend_discount(prob, base=5, max_disc=20):
    if prob >= 0.75: return base
    if prob >= 0.60: return min(base+5, max_disc)
    if prob >= 0.45: return min(base+10, max_disc)
    return min(base+15, max_disc)

@app.get("/")
def home():
    return {"status": "up"}

@app.post("/score")
def score(rec: Record):
    row = pd.DataFrame([rec.data], columns=feats)
    prob = float(model.predict_proba(row)[0,1])
    return {
        "predicted_purchase_prob": prob,
        "recommended_discount": recommend_discount(prob)
    }
