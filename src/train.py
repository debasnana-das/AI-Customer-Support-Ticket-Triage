from pathlib import Path
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'tickets.csv'
MODELS = ROOT / 'models'
MODELS.mkdir(exist_ok=True)

def train_target(df, target, name):
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df[target], test_size=0.2, random_state=42, stratify=df[target]
    )
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=1, sublinear_tf=True)
    Xtr = vec.fit_transform(X_train)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf.fit(Xtr, y_train)
    joblib.dump({'vectorizer': vec, 'model': clf}, MODELS / f'{name}.joblib')
    pd.DataFrame({'text': X_test, 'label': y_test}).to_csv(ROOT/'outputs'/f'{name}_test.csv', index=False)

if not DATA.exists():
    raise FileNotFoundError('Run data/generate_dataset.py first.')
df = pd.read_csv(DATA)
train_target(df, 'category', 'category_model')
train_target(df, 'urgency', 'urgency_model')
print('Training complete.')
