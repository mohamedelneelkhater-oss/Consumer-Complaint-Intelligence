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
DATA_PATH = BASE_DIR / "test_df.csv" if (BASE_DIR / "test_df.csv").exists() else (BASE_DIR / "data" / "sample_complaints.csv")

st.set_page_config(page_title="Consumer Complaint Classifier", page_icon="📄", layout="wide")

def pretty_label(label: str) -> str:
    return str(label).replace("_", " ").title()

@st.cache_resource
def load_system():
    """Load model, tokenizer, label names and NLP resources once (Robust path search)."""
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    possible_dirs = [BASE_DIR, BASE_DIR / "model", BASE_DIR / "files"]
    
    model_path, tokenizer_path, label_path, info_path = None, None, None, None
    
    for d in possible_dirs:
        if (d / "best_model.keras").exists():
            model_path = d / "best_model.keras"
        elif (d / "best_keras_model.h5").exists():
            model_path = d / "best_keras_model.h5"
            
        if (d / "keras_tokenizer.pkl").exists():
            tokenizer_path = d / "keras_tokenizer.pkl"
            
        if (d / "label_names.json").exists():
            label_path = d / "label_names.json"
            
        if (d / "model_info.json").exists():
            info_path = d / "model_info.json"

    if not model_path:
        st.error("❌ Model file ('best_model.keras' or 'best_keras_model.h5') not found in any directory!")
        st.stop()
        
    model = load_model(str(model_path))
    
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
        
    with open(label_path) as f:
        label_names = json.load(f)
        
    if info_path and info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
    else:
        info = {"best_model": "GRU", "max_len": 150}

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

# Load 10 diverse samples representing different classes from dataset
diverse_samples = []
if DATA_PATH.exists():
    df_temp = pd.read_csv(DATA_PATH)
    target_col = 'clean_text' if 'clean_text' in df_temp.columns else ('text' if 'text' in df_temp.columns else df_temp.columns[0])
    label_col = 'label' if 'label' in df_temp.columns else (df_temp.columns[1] if len(df_temp.columns) > 1 else None)
    
    if label_col and label_col in df_temp.columns:
        for _, group in df_temp.groupby(label_col):
            texts = group[target_col].dropna().astype(str).tolist()
            if texts:
                diverse_samples.append(texts[0])
    if not diverse_samples:
        diverse_samples = df_temp[target_col].dropna().astype(str).tolist()
    diverse_samples = diverse_samples[:10]
else:
    diverse_samples = [
        "There is an account on my credit report that I never opened.",
        "A collection agency keeps calling my workplace about a debt I do not owe.",
        "I was charged an annual fee and interest on my credit card unexpectedly.",
        "My checking account was frozen without notice and I could not access my money.",
        "My mortgage servicer miscalculated my escrow account and payments increased."
    ]

# Main UI Header
st.caption("FINANCIAL INTELLIGENCE · NLP ENGINE")
st.title("Consumer Complaint Classifier")
st.write(f"Real-time complaint categorization powered by **{info.get('best_model', 'GRU')}**.")

tab_single, tab_batch = st.tabs(["Single Complaint", "Batch Classification & Save"])

# ------------------------------------------------------------ Single Complaint Tab
with tab_single:
    st.subheader("Analyze Single Complaint")
    
    choice = st.selectbox(
        "Select a diverse sample text from dataset classes", 
        ["-- write your own or select a sample --"] + diverse_samples
    )
    
    default_text = choice if choice != "-- write your own or select a sample --" else ""
    complaint = st.text_area("Complaint narrative", value=default_text, height=160)

    if st.button("Classify complaint", type="primary"):
        if not complaint.strip():
            st.warning("Please enter a complaint narrative first.")
        else:
            probs = predict_proba([complaint])[0]
            best = int(np.argmax(probs))
            st.success(f"**Predicted Category:** {pretty_label(label_names[best])}  |  **Confidence:** {probs[best]:.1%}")
            
            st.markdown("#### Category Probabilities Breakdown")
            for idx in np.argsort(probs)[::-1]:
                st.text(f"{pretty_label(label_names[idx])}: {probs[idx]:.1%}")
                st.progress(float(probs[idx]))

# ------------------------------------------------------------ Batch Tab with Auto-Save
with tab_batch:
    st.subheader("Batch Process & Auto-Save to test_df.csv")
    st.write("Processes clean text, appends prediction columns, and saves directly via `test_df.to_csv('test_df.csv', index=False)`.")

    if st.button("Run Batch Classification & Save", type="primary"):
        if DATA_PATH.exists():
            with st.spinner("Classifying and saving..."):
                test_df = pd.read_csv(DATA_PATH).head(30)
                target_col = 'clean_text' if 'clean_text' in test_df.columns else ('text' if 'text' in test_df.columns else test_df.columns[0])
                
                texts = test_df[target_col].astype(str).tolist()
                probs = predict_proba(texts)
                best_idx = probs.argmax(axis=1)
                
                test_df['Predicted_Class'] = [pretty_label(label_names[i]) for i in best_idx]
                test_df['Confidence_Score'] = [f"{probs[r, i]:.1%}" for r, i in enumerate(best_idx)]
                
                # Save test_df back to csv with index=False
                test_df.to_csv("test_df.csv", index=False)
                print("✅ Saved test_df.csv successfully!")
                
                st.success("✅ Batch classification complete! test_df.csv saved successfully with predictions.")
                st.dataframe(test_df)
        else:
            st.warning("`test_df.csv` file not found in the workspace.")

st.markdown("---")
st.caption("Consumer Complaint Intelligence Platform · Production System")
