from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
CATEGORY = joblib.load(ROOT/'models/category_model.joblib')
URGENCY = joblib.load(ROOT/'models/urgency_model.joblib')

QUEUE_MAP = {
    'billing': 'Billing Support',
    'technical': 'Technical Support',
    'account': 'Account Support',
    'product': 'Product Support',
}

def _predict(bundle, text):
    x = bundle['vectorizer'].transform([text])
    model = bundle['model']
    pred = model.predict(x)[0]
    probs = model.predict_proba(x)[0]
    conf = float(max(probs))
    return pred, conf

def predict_ticket(text: str, threshold: float = 0.60):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Ticket text must be a non-empty string.')
    category, c_conf = _predict(CATEGORY, text)
    urgency, u_conf = _predict(URGENCY, text)
    review = min(c_conf, u_conf) < threshold
    return {
        'category': category,
        'urgency': urgency,
        'queue': QUEUE_MAP.get(category, 'General Support'),
        'category_confidence': round(c_conf, 4),
        'urgency_confidence': round(u_conf, 4),
        'human_review_required': review,
    }
