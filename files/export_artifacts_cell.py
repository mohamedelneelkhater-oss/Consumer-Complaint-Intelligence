# Paste this as the LAST cell of the Kaggle notebook (replaces section 12).
# It creates every file the Streamlit app needs: model/ and data/ folders.
import os, json, pickle
import tensorflow as tf

os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

# 1. Pick the best recurrent model from the benchmark table
models_by_name = {"SimpleRNN": simple_rnn, "LSTM": lstm_model, "GRU": gru_model}
best_model_name = results_df.iloc[0]["model"]
best_model = models_by_name[best_model_name]
best_model.save("model/best_model.keras")

# 2. Tokenizer
with open("model/keras_tokenizer.pkl", "wb") as f:
    pickle.dump(keras_tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

# 3. Label names in the EXACT order the model outputs probabilities
with open("model/label_names.json", "w") as f:
    json.dump(label_names, f)

# 4. Model info shown in the app sidebar
best_row = results_df.iloc[0]
with open("model/model_info.json", "w") as f:
    json.dump({
        "best_model": best_model_name,
        "max_len": MAX_LEN,
        "accuracy": float(best_row["accuracy"]),
        "f1_score": float(best_row["f1_score"]),
    }, f)

# 5. Small raw-text sample for the batch tab (50 rows)
test_df[["text", "label"]].sample(50, random_state=SEED).to_csv("data/sample_complaints.csv", index=False)

print("Best model:", best_model_name)
print("Label order:", label_names)
print("TensorFlow version:", tf.__version__, "(use this in requirements.txt)")
