"""Streamlit interface for the movie review sentiment classifier.
The prediction comes ONLY from the trained Logistic Regression model."""
import json
import random
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from preprocess import clean_text

MODEL_DIR = Path(__file__).resolve().parent / "models"
CLOSE_CALL = 0.65   # below this confidence we warn the user

EXAMPLES = [
    "An absolutely brilliant film. The acting was wonderful and the story kept me hooked until the very end.",
    "This was a boring, badly written mess. The plot made no sense and I want my money back.",
    "The movie was not good at all. I expected much more from this director.",
    "Beautiful cinematography, but the pacing was slow and the ending felt disappointing.",
]

st.set_page_config(page_title="Movie Review Sentiment Analyzer", page_icon="🎬")


# ---------- Load the saved model once ----------
@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_DIR / "logistic_regression.joblib")
    vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    return model, vectorizer, vectorizer.get_feature_names_out(), metrics


if not (MODEL_DIR / "logistic_regression.joblib").exists():
    st.error("Saved model not found. Run `python train_model.py` first.")
    st.stop()

model, vectorizer, feature_names, metrics = load_resources()


# ---------- Prediction ----------
def analyze(review: str) -> dict:
    cleaned = clean_text(review)
    features = vectorizer.transform([cleaned])
    probs = model.predict_proba(features)[0]          # [negative, positive]
    pred = int(np.argmax(probs))

    # Which words pushed the score? (TF-IDF value x learned weight)
    row = features.tocoo()
    coefs = model.coef_[0]
    contributions = [(feature_names[j], float(v * coefs[j]))
                     for j, v in zip(row.col, row.data)]
    contributions.sort(key=lambda x: x[1])
    return {
        "review": review,
        "cleaned": cleaned,
        "label": "Positive" if pred == 1 else "Negative",
        "confidence": float(probs[pred]),
        "probs": probs,
        "toward_negative": contributions[:5],
        "toward_positive": contributions[::-1][:5],
    }


# ---------- Session state ----------
st.session_state.setdefault("review_text", "")
st.session_state.setdefault("result", None)
st.session_state.setdefault("history", [])


def load_example():
    st.session_state.review_text = random.choice(EXAMPLES)


def clear_all():
    st.session_state.review_text = ""
    st.session_state.result = None


# ---------- Header ----------
st.title("🎬 Movie Review Sentiment Analyzer")
st.write(
    "Type or paste a movie review. A machine learning model (TF-IDF + Logistic "
    "Regression) trained on 50,000 IMDb reviews will predict whether the "
    "sentiment is **positive** or **negative**."
)

st.text_area("Movie review", key="review_text", height=170,
             placeholder="e.g. The acting was superb and the story was moving...")

c1, c2, c3 = st.columns(3)
analyze_clicked = c1.button("🔍 Analyze Review", type="primary", use_container_width=True)
c2.button("🎲 Example Review", on_click=load_example, use_container_width=True)
c3.button("🧹 Clear", on_click=clear_all, use_container_width=True)

if analyze_clicked:
    text = st.session_state.review_text.strip()
    if not text:
        st.warning("Please enter a review first.")
    elif not clean_text(text):
        st.warning("No usable words found in that text. Try a longer review.")
    else:
        result = analyze(text)
        st.session_state.result = result
        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Review": text[:80] + ("..." if len(text) > 80 else ""),
            "Sentiment": result["label"],
            "Confidence": f"{result['confidence']:.1%}",
        })

# ---------- Results ----------
res = st.session_state.result
if res:
    st.divider()
    color, icon = ("#16a34a", "😊") if res["label"] == "Positive" else ("#dc2626", "😞")
    left, right = st.columns(2)

    with left:
        st.markdown(
            f"""<div style="padding:1.2rem;border-radius:12px;border:2px solid {color};
            background:{color}22;text-align:center;">
            <div style="font-size:3rem">{icon}</div>
            <div style="font-size:1.8rem;font-weight:700;color:{color}">{res['label']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.metric("Model confidence", f"{res['confidence']:.1%}")
        st.progress(res["confidence"])
        st.caption("This is the model's estimated probability, not certainty. "
                   "The model can be wrong.")
        if res["confidence"] < CLOSE_CALL:
            st.info("⚠️ Close call: the model found this review fairly mixed or unclear.")

    with right:
        st.write("**Class probabilities**")
        chart = pd.DataFrame({"Probability": res["probs"]},
                             index=["Negative", "Positive"])
        st.bar_chart(chart)

    with st.expander("How was this prediction generated?"):
        st.markdown(
            "1. Your text is cleaned (lowercase, HTML and punctuation removed, "
            "stop words dropped, negations like *not* kept).\n"
            "2. The cleaned text is turned into TF-IDF numbers using the saved vectorizer.\n"
            "3. The Logistic Regression model combines those numbers with the word "
            "weights it learned during training and outputs a probability for each class.\n"
        )
        st.write("**Cleaned text the model saw:**")
        st.code(res["cleaned"], language=None)
        st.write("**Words and phrases that influenced the score most** "
                 "(weight × TF-IDF; positive pushes toward *Positive*, negative toward *Negative*):")
        a, b = st.columns(2)
        a.write("Toward Positive")
        a.dataframe(pd.DataFrame(res["toward_positive"], columns=["Word", "Influence"]).round(3),
                    hide_index=True)
        b.write("Toward Negative")
        b.dataframe(pd.DataFrame(res["toward_negative"], columns=["Word", "Influence"]).round(3),
                    hide_index=True)
        st.caption("Features can be single words or word pairs (e.g. \"not good\"). "
                   "This shows the model's learned weights, which is a simplified view of its behaviour.")

# ---------- History ----------
st.divider()
st.subheader("Review history (this session)")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True,
                 use_container_width=True)
else:
    st.caption("No reviews analyzed yet.")

# ---------- Evaluation metrics ----------
st.divider()
st.subheader("Model evaluation")
st.caption(f"Measured on {metrics['test_size']:,} held-out test reviews "
           f"(training set: {metrics['train_size']:,}). This app uses Logistic Regression; "
           f"the best model by F1-score was {metrics['best_model']}.")
table = pd.DataFrame(metrics["results"]).T[["accuracy", "precision", "recall", "f1"]]
table.columns = ["Accuracy", "Precision", "Recall", "F1-score"]
st.dataframe(table.round(4), use_container_width=True)

with st.expander("Confusion matrices"):
    for name in metrics["results"]:
        img = MODEL_DIR / f"confusion_{name.lower().replace(' ', '_')}.png"
        if img.exists():
            st.image(str(img), caption=name, width=380)

st.caption("Limitations: the model only knows positive and negative (the IMDb dataset has no "
           "neutral label), is trained on English movie reviews, and can misread sarcasm.")