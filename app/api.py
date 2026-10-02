from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
from inference import predict_ticket

app = FastAPI(title='AI Customer Support Ticket Triage', version='1.0')

class Ticket(BaseModel):
    text: str
    confidence_threshold: float = 0.60

@app.get('/')
def root():
    return {'project': 'AI Customer Support Ticket Triage', 'status': 'ready'}

@app.post('/predict')
def predict(item: Ticket):
    try:
        return predict_ticket(item.text, item.confidence_threshold)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
