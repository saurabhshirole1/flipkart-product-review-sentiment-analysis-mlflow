import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# ---------------- NLTK Setup ----------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------------- Load Model & Vectorizer ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_DIR = os.path.join(BASE_DIR, "..", "pkl")

with open(os.path.join(PKL_DIR, "sentiment_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(PKL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    tfidf = pickle.load(f)

# ---------------- Text Cleaning ----------------
def clean_review(text):
    text = str(text)
    text = text.replace('READ MORE', '').replace('read more', '')
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.markdown(
    """
    <h1 style="white-space: nowrap; text-align: center;">
        🛒 Flipkart Review Sentiment Analysis
    </h1>
    """,
    unsafe_allow_html=True
)


review = st.text_area(
    "✍️ Enter a product review:",
    placeholder="Example: The product quality is excellent and delivery was super fast!"
)

if st.button("🚀 Predict Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        cleaned = clean_review(review)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.markdown(
                "<h2 style='color:#2ecc71; text-align:center;'>✅ Positive Review </h3>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<h2 style='color:#e74c3c; text-align:center;'>❌ Negative Review </h3>",
                unsafe_allow_html=True
            )
