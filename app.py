import streamlit as st
import joblib
from utils import clean_text

# Load trained model & vectorizer
model = joblib.load("models/logistic_model.pkl")
tfidf = joblib.load("models/tfidf.pkl")

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection")

user_input = st.text_area("Enter a news article text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vec = tfidf.transform([cleaned])
        prediction = model.predict(vec)[0]
        st.success("✅ Real News" if prediction == 0 else "❌ Fake News")
