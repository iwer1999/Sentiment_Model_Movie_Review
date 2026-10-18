"""Shows the weight the Logistic Regression model gives to words."""
import sys
from pathlib import Path
import joblib

MODEL_DIR = Path(__file__).resolve().parent / "models"
model = joblib.load(MODEL_DIR / "logistic_regression.joblib")
vocab = joblib.load(MODEL_DIR / "vectorizer.joblib").vocabulary_

for word in sys.argv[1:]:
    if word in vocab:
        weight = model.coef_[0][vocab[word]]
        print(f"{word:12} weight {weight:+.3f}  ({'positive' if weight > 0 else 'negative'})")
    else:
        print(f"{word:12} NOT in vocabulary (the model ignores it)")