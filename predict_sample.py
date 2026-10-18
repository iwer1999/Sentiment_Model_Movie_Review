"""Quick check that the saved model works on new text."""
from pathlib import Path
import json
import joblib
from preprocess import clean_text

MODEL_DIR = Path(__file__).resolve().parent / "models"
summary = json.loads((MODEL_DIR / "metrics.json").read_text())
file_stub = summary["best_model"].lower().replace(" ", "_")

model = joblib.load(MODEL_DIR / f"{file_stub}.joblib")
vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")

tests = [
    "An absolutely brilliant film with wonderful acting. I loved every minute.",
    "This was a boring, badly written mess. I want my money back.",
    "The movie was not good at all.",
]
for review in tests:
    features = vectorizer.transform([clean_text(review)])
    label = "Positive" if model.predict(features)[0] == 1 else "Negative"
    print(f"{label:9} <- {review}")