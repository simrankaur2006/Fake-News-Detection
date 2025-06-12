import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Constants
MAX_LEN = 300
MAX_WORDS = 10000

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("fake_news_bilstm_glove.h5")
    return model

@st.cache(allow_output_mutation=True)
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

model = load_model()
tokenizer = load_tokenizer()

# UI layout
st.set_page_config(page_title="🧠 Fake News Detector", layout="centered")
st.title("🧠 Fake News Detector")
st.write("Enter a news article or headline to detect whether it's **Fake** or **Real** using a deep learning model (BiLSTM + GloVe).")

user_input = st.text_area("📰 Paste the news text below:", height=200)

if st.button("🔍 Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocess input
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

        # Predict
        prediction = model.predict(padded)[0][0]
        real_confidence = prediction
        fake_confidence = 1 - prediction

        # Output result
        if prediction >= 0.5:
            st.success(f"✅ This looks like **REAL** news (Confidence: {real_confidence:.2f})")
        else:
            st.error(f"❌ This looks like **FAKE** news (Confidence: {fake_confidence:.2f})")

        # Pie chart
        fig, ax = plt.subplots()
        ax.pie([fake_confidence, real_confidence], labels=["Fake", "Real"], autopct="%1.1f%%",
               colors=["#FF6B6B", "#4CAF50"], startangle=140)
        ax.axis("equal")
        st.pyplot(fig)