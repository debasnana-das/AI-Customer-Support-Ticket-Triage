from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

QUEUE_MAP = {
    "billing": "Billing Support",
    "technical": "Technical Support",
    "account": "Account Support",
    "product": "Product Support",
}

PATTERNS = {
    "billing": [
        "I was charged twice for my subscription",
        "my invoice has an incorrect amount",
        "please explain this payment on my card",
        "I need a refund for the last billing cycle",
        "the monthly charge is higher than expected",
        "my discount was not applied to the invoice",
    ],
    "technical": [
        "the application crashes when I upload a file",
        "the website shows an error after login",
        "the API returns a server error intermittently",
        "the mobile app freezes on the checkout page",
        "I cannot connect to the service from my laptop",
        "the dashboard is loading forever",
    ],
    "account": [
        "I cannot reset my password",
        "my account is locked after several login attempts",
        "please change the email address on my account",
        "I need help verifying my identity",
        "I forgot which phone number is linked to my account",
        "my profile details are incorrect",
    ],
    "product": [
        "how do I use the new reporting feature",
        "the export option is missing from my plan",
        "I want to know whether this feature is supported",
        "please explain how the workflow automation works",
        "the search feature does not return expected results",
        "I need guidance on configuring the product",
    ],
}

URGENCY_PHRASES = {
    "high": ["urgent", "immediately", "critical", "blocked", "down", "cannot work", "production impact"],
    "medium": ["today", "important", "soon", "deadline", "affecting my work"],
    "low": ["when possible", "just a question", "no rush", "for information", "whenever convenient"],
}

def _build_training_data():
    rows = []
    for category, examples in PATTERNS.items():
        for text in examples:
            for phrase in URGENCY_PHRASES["low"]:
                rows.append((f"{text}. This is {phrase} for me.", category, "low"))
            for phrase in URGENCY_PHRASES["medium"]:
                rows.append((f"{text}. This is {phrase} for me.", category, "medium"))
            for phrase in URGENCY_PHRASES["high"]:
                rows.append((f"{text}. This is {phrase} for me.", category, "high"))
    return pd.DataFrame(rows, columns=["text", "category", "urgency"])

def _train_bundle(df, target):
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
    X = vectorizer.fit_transform(df["text"])
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X, df[target])
    return {"vectorizer": vectorizer, "model": model}

def _ensure_models():
    category_path = MODELS / "category_model.joblib"
    urgency_path = MODELS / "urgency_model.joblib"
    if category_path.exists() and urgency_path.exists():
        return
    df = _build_training_data()
    joblib.dump(_train_bundle(df, "category"), category_path)
    joblib.dump(_train_bundle(df, "urgency"), urgency_path)

_ensure_models()
CATEGORY = joblib.load(MODELS / "category_model.joblib")
URGENCY = joblib.load(MODELS / "urgency_model.joblib")

def _predict(bundle, text):
    x = bundle["vectorizer"].transform([text])
    model = bundle["model"]
    pred = model.predict(x)[0]
    probs = model.predict_proba(x)[0]
    conf = float(max(probs))
    return pred, conf

def predict_ticket(text: str, threshold: float = 0.60):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Ticket text must be a non-empty string.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1.")

    category, c_conf = _predict(CATEGORY, text)
    urgency, u_conf = _predict(URGENCY, text)
    review = min(c_conf, u_conf) < threshold

    return {
        "category": category,
        "urgency": urgency,
        "queue": QUEUE_MAP.get(category, "General Support"),
        "category_confidence": round(c_conf, 4),
        "urgency_confidence": round(u_conf, 4),
        "human_review_required": review,
    }
