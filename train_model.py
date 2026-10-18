"""Loads IMDb, preprocesses, trains 3 models, evaluates, and saves the results."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")                      # draw to files, no pop-up windows
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from preprocess import clean_text

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "aclImdb"
MODEL_DIR = BASE_DIR / "models"
RANDOM_STATE = 42                           # makes the split repeatable
LABEL_NAMES = ["Negative", "Positive"]      # 0 = negative, 1 = positive


def load_reviews() -> pd.DataFrame:
    """Reads every labelled review (train + test folders) into one table."""
    rows = []
    for split in ["train", "test"]:
        for folder_name, label in [("neg", 0), ("pos", 1)]:
            folder = DATA_DIR / split / folder_name
            for file in sorted(folder.glob("*.txt")):
                rows.append((file.read_text(encoding="utf-8"), label))
    return pd.DataFrame(rows, columns=["review", "sentiment"])


def main():
    if not DATA_DIR.exists():
        raise SystemExit("Dataset not found. Run: python download_data.py")
    MODEL_DIR.mkdir(exist_ok=True)

    # ---- 1. Load and inspect ------------------------------------------
    df = load_reviews()
    print("Total reviews:", len(df))
    print("Label counts (0=negative, 1=positive):")
    print(df["sentiment"].value_counts())
    print("\nExample review:\n", df["review"].iloc[0][:300], "...")
    print("Average review length (words):", round(df["review"].str.split().str.len().mean()))

    # ---- 2. Preprocess -------------------------------------------------
    print("\nCleaning text (about a minute)...")
    df["clean_review"] = df["review"].apply(clean_text)

    # ---- 3. Train/test split (80/20, same ratio of pos/neg in each) -----
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_review"], df["sentiment"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["sentiment"])
    print(f"Training reviews: {len(X_train)} | Testing reviews: {len(X_test)}")

    # ---- 4. TF-IDF features --------------------------------------------
    # Fit on TRAINING data only, so the test set stays truly unseen.
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                 min_df=2, sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print("TF-IDF matrix shape (reviews x features):", X_train_tfidf.shape)

    # ---- 5. Train and evaluate three models ------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(),
    }
    results = {}
    for name, model in models.items():
        print(f"\n===== {name} =====")
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)

        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "precision": round(float(precision_score(y_test, preds)), 4),
            "recall": round(float(recall_score(y_test, preds)), 4),
            "f1": round(float(f1_score(y_test, preds)), 4),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }
        print(classification_report(y_test, preds, target_names=LABEL_NAMES))
        print("Confusion matrix:\n", confusion_matrix(y_test, preds))

        # Save the model and a confusion-matrix picture
        file_stub = name.lower().replace(" ", "_")
        joblib.dump(model, MODEL_DIR / f"{file_stub}.joblib")
        ConfusionMatrixDisplay.from_predictions(
            y_test, preds, display_labels=LABEL_NAMES, cmap="Blues")
        plt.title(name)
        plt.savefig(MODEL_DIR / f"confusion_{file_stub}.png", bbox_inches="tight")
        plt.close()

    # ---- 6. Pick the best model by F1-score on the test set ---------------
    best_name = max(results, key=lambda n: results[n]["f1"])
    print(f"\nBest model by F1-score: {best_name}")

    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.joblib")
    summary = {
        "best_model": best_name,
        "random_state": RANDOM_STATE,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "results": results,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("Saved models, vectorizer and metrics to the 'models' folder.")


if __name__ == "__main__":
    main()