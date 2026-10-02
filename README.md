# AI Customer Support Ticket Triage

An end-to-end AI-powered customer support ticket triage system that classifies incoming support tickets by category and urgency, routes them to appropriate support queues, and flags low-confidence predictions for human review.

## Objective
Read an incoming support ticket, predict its **category** and **urgency**, and route it to the corresponding queue.

## Categories
- billing
- technical
- account
- product

## Urgency
- low
- medium
- high

## Architecture
`raw ticket -> TF-IDF -> category classifier + urgency classifier -> confidence check -> queue routing`

## Project Components
- `data/generate_dataset.py` — creates a synthetic labeled dataset for development
- `src/train.py` — trains the category and urgency models
- `src/inference.py` — predicts category, urgency, queue, and confidence
- `src/evaluate.py` — evaluates both models on held-out test data
- `app/api.py` — FastAPI service
- `app/streamlit_app.py` — Streamlit interface
- `tests/test_inference.py` — smoke test

## Installation
```bash
pip install -r requirements.txt
```

## Training and Evaluation
```bash
python data/generate_dataset.py
python src/train.py
python src/evaluate.py
```

## Run the API
```bash
uvicorn app.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

## Run the Streamlit UI
```bash
streamlit run app/streamlit_app.py
```

## Human Review
The default confidence threshold is `0.60`. When either the category or urgency prediction falls below the threshold, the ticket is flagged for human review.

## Development Data
The repository contains a synthetic dataset generated for development and demonstration purposes.
