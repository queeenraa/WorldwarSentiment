import streamlit as st
import joblib
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load model dan vectorizer
model = joblib.load('model_nb.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Stopwords (Indonesia + English)
stop_words = set(stopwords.words('indonesian') + stopwords.words('english'))

# Fungsi pre-processing teks
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^\w\s]", "", text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# UI Streamlit
st.title("🗣️ Sentiment Analysis World Issue Tweets")

user_input = st.text_area("Masukkan teks tweet di sini:", "")

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Tolong masukkan teks terlebih dahulu.")
    else:
        cleaned = clean_text(user_input)
        tfidf_input = vectorizer.transform([cleaned])
        prediction = model.predict(tfidf_input)[0]
        st.success(f"🎯 Prediksi Sentimen: **{prediction.upper()}**")
