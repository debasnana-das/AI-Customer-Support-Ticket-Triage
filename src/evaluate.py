from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
for name, target in [("category_model", "category"), ("urgency_model", "urgency")]:
    bundle = joblib.load(ROOT / "models" / f"{name}.joblib")
    test = pd.read_csv(ROOT / "outputs" / f"{name}_test.csv")
    x = bundle["vectorizer"].transform(test["text"])
    pred = bundle["model"].predict(x)
    print(f"\n=== {target.upper()} (HELD-OUT TEST SET) ===")
    print(classification_report(test["label"], pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(test["label"], pred))
