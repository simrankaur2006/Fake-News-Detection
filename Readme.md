# 📰 Fake News Detection using Machine Learning

This project is a Fake News and Scam Detection System built using Machine Learning (Logistic Regression). It analyzes news headlines or article texts to determine whether the content is real or fake, helping users identify misinformation and scams effectively.

The system uses Natural Language Processing (NLP) techniques such as TF-IDF vectorization to convert text data into numerical form, and a Logistic Regression classifier to make predictions.
A simple and interactive Streamlit web app is provided for users to input text and get instant results.


# ⚙️Features

Detects fake or scam news headlines/articles

Built using Logistic Regression (a supervised ML algorithm)

TF-IDF Vectorizer for text feature extraction

Streamlit interface for easy web-based predictions

Lightweight and fast — runs locally or can be deployed online


# Check the live web interface 
[https://fake-news-detection-simran.streamlit.app/] live link

Deployed on streamlit web interface using streamlit cloud


## 🚀 Technologies Used
- Python
- Scikit-learn
- Streamlit
- Pickle
- Pandas, Numpy
- NLP Techniques


# Project 
Fake-News-Detection/
│
├── app.py                # Streamlit app file (web interface)

├── model.pkl             # Saved Machine Learning model

├── vectorizer.pkl        # Saved vectorizer

├── requirements.txt      # List of dependencies

├── README.md             # Description of  project

└── train.csv             # dataset used
  
## 🧠 Model
Trained using NLP Techniques such as TF-IDF vectorization and Machine Learning algorithm like Logistic Regression.

## ▶️ How to Run
```bash
 git clone
 cd Fake-News-Detection
 pip install -r requirements.txt
 streamlit run app.py

