"""Consumer Complaint Intelligence - Streamlit app (GRU model)."""
import json
import pickle
import re
import string
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_PATH = BASE_DIR / "data" / "sample_complaints.csv"

st.set_page_config(page_title="Consumer Complaint Classifier", page_icon="📄", layout="wide")

EXAMPLES = {
    "Credit reporting": "There is an account on my credit report that I never opened. I disputed it with the bureau three times and they still have not removed it.",
    "Debt collection": "A collection agency keeps calling my workplace about a debt I do not owe, even after I asked them in writing to stop contacting me.",
    "Credit card": "I was charged an annual fee and interest on my credit card even though the promotional offer promised no fees for the first year.",
    "Retail banking": "My checking account was frozen without notice and I could not access my money for over a week. The bank gave no explanation.",
    "Mortgages and loans": "My mortgage servicer miscalculated my escrow account and my monthly payment suddenly increased by hundreds of dollars.",
}


def pretty_label(label: str) -> str:
    return str(label).replace("_", " ").title()


@st.cache_resource(show_spinner="Loading model...")
def load_system():
    """Load model, tokenizer, label names and NLP resources once."""
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    model = load_model(MODEL_DIR / "best_model.keras")
    with open(MODEL_DIR / "keras_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open(MODEL_DIR / "label_names.json") as f:
        label_names = json.load(f)
    with open(MODEL_DIR / "model_info.json") as f:
        info = json.load(f)

    return model, tokenizer, label_names, info, WordNetLemmatizer(), set(stopwords.words("english"))


model, tokenizer, label_names, info, lemmatizer, stop_words = load_system()
MAX_LEN = int(info.get("max_len", 150))


def clean_text(text: str) -> str:
    """Same cleaning pipeline used during training."""
    text = str(text).lower()
    text = re.sub(r"x{2,}", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\d+", "", text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)


def predict_proba(texts):
    cleaned = [clean_text(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    return model.predict(padded, batch_size=512, verbose=0)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Model")
    st.write(f"**Architecture:** {info.get('best_model', 'GRU')}")
    if "f1_score" in info:
        st.write(f"**Test F1 (weighted):** {info['f1_score']:.2%}")
    if "accuracy" in info:
        st.write(f"**Test accuracy:** {info['accuracy']:.2%}")
    st.write("**Categories:**")
    for name in label_names:
        st.write(f"- {pretty_label(name)}")

tab_single, tab_batch = st.tabs(["Single complaint", "Batch classification"])

# ------------------------------------------------------------ single tab
with tab_single:
    st.caption("FINANCIAL INTELLIGENCE · NLP ENGINE")
    st.title("Consumer Complaint Classifier")
    st.write("Paste a customer complaint and get the predicted product category with confidence.")

    choice = st.selectbox("Try an example", ["-- write your own --"] + list(EXAMPLES))
    default_text = EXAMPLES.get(choice, "")
    complaint = st.text_area("Complaint narrative", value=default_text, height=160, key=f"text_{choice}")

    if st.button("Classify complaint", type="primary"):
        if not complaint.strip():
            st.warning("Please enter a complaint narrative first.")
        else:
            probs = predict_proba([complaint])[0]
            best = int(np.argmax(probs))
            st.success(f"**Predicted category:** {pretty_label(label_names[best])}  |  **Confidence:** {probs[best]:.1%}")
            st.markdown("#### Category probabilities")
            for idx in np.argsort(probs)[::-1]:
                st.write(f"{pretty_label(label_names[idx])} ({probs[idx]:.1%})")
                st.progress(float(probs[idx]))

# ------------------------------------------------------------ batch tab
with tab_batch:
    st.title("Batch classification")
    st.write("Classify many complaints at once, using the bundled sample or your own CSV file.")

    uploaded = st.file_uploader("Upload a CSV file (optional)", type="csv")
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
    elif SAMPLE_PATH.exists():
        batch_df = pd.read_csv(SAMPLE_PATH)
        st.info("Using the bundled sample file (`data/sample_complaints.csv`).")
    else:
        batch_df = None
        st.warning("No sample file found. Upload a CSV to continue.")

    if batch_df is not None and not batch_df.empty:
        text_cols = [c for c in batch_df.columns if batch_df[c].dtype == object]
        default_idx = text_cols.index("text") if "text" in text_cols else 0
        text_col = st.selectbox("Text column", text_cols, index=default_idx)

        if st.button("Run batch classification", type="primary"):
            with st.spinner("Classifying..."):
                probs = predict_proba(batch_df[text_col].astype(str).tolist())
            best_idx = probs.argmax(axis=1)
            result = batch_df.copy()
            result["predicted_category"] = [pretty_label(label_names[i]) for i in best_idx]
            result["confidence"] = [f"{probs[r, i]:.1%}" for r, i in enumerate(best_idx)]

            if "label" in result.columns:
                correct = (result["label"].astype(str) == np.array(label_names)[best_idx]).mean()
                st.metric("Accuracy on this file", f"{correct:.1%}")

            st.dataframe(result)
            st.download_button(
                "Download results (CSV)",
                result.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )

st.markdown("---")
st.caption("Consumer Complaint Intelligence · Instant course project")
