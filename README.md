# Multilingual-Formality-Detector-API

Fine-tuned formality detector for 4 languages (French, English, German, Spanish), realized using XLM-RoBERTa.

## Prerequisites

Install requirements

pip install -r requirements.txt

## Test formation

python tests.py

## Training

python XLMRoberta.py

## API Running

uvicorn app:app --reload

Visit http://127.0.0.1:8000/docs
