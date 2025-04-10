from transformers import pipeline
import torch
from fastapi import FastAPI
import pandas as pd
import os

app = FastAPI()

classifier = pipeline(
    "text-classification",
    model="./xlm-r-formality-multilingual",
    tokenizer="./xlm-r-formality-multilingual",
    device=0 if torch.cuda.is_available() else -1
)

@app.get("/")
def home():
    return {"message": "Formality Classifier API is running!"}

@app.post("/predict")
def predict(text: str):
    result = classifier(text)
    return {"text": text, "prediction": result[0]}