from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import detect_toxicity

app = FastAPI()

class TextIn(BaseModel):
    text: str

class DetectionOut(BaseModel):
    toxicity: str

@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/detect", response_model=DetectionOut)
def detect(payload: TextIn):
    toxicity = detect_toxicity(payload.text)
    return {"toxicity": toxicity}