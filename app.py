import streamlit as st
import pickle

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the vectorizer (note: filename is 'vector.pkl')
with open("vector.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit app UI
st.title("📰 Fake News Detection (Scam Detection)")
input_text = st.text_input("Enter news headline or article...")

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter some text before predicting.")
    else:
        # Transform the input using the vectorizer
        processed_input = vectorizer.transform([input_text])
        # Make prediction
        prediction = model.predict(processed_input)[0]
        # Display result
        if prediction == 1:
            st.success("✅ This news seems to be **Real**.")
        else:
            st.error("🚨 This news seems to be **Fake**.")
