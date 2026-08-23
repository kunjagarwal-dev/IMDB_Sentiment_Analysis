import streamlit as st
import json
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------- Page Config ----------
st.set_page_config(
    page_title="Movie Review Sentiment",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------- Custom CSS ----------
st.markdown("""
    <style>
    .title-text {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #f59e0b, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .subtitle-text {
        text-align: center;
        color: #9ca3af;
        font-size: 16px;
        margin-bottom: 30px;
    }
    .prediction-box {
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin-top: 15px;
    }
    .positive-box {
        background: linear-gradient(135deg, #10b981, #059669);
    }
    .negative-box {
        background: linear-gradient(135deg, #ef4444, #b91c1c);
    }
    .prediction-label {
        font-size: 36px;
        font-weight: 900;
        color: white;
        margin: 0;
    }
    .confidence-text {
        color: #f3f4f6;
        font-size: 16px;
        margin-top: 5px;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #f59e0b, #ec4899);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
    }
    div.stButton > button:hover {
        opacity: 0.9;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

VOCAB_SIZE = 10000
MAXLEN = 200
INDEX_FROM = 3  # Keras IMDB reserves 0-3 for special tokens

# ---------- Load Model and Word Index ----------
@st.cache_resource
def get_model():
    return load_model("models/imdb_gru_model.h5")

@st.cache_resource
def get_word_index():
    with open("models/imdb_word_index.json") as f:
        word_index = json.load(f)
    # Keras's IMDB word_index needs offsetting to match load_data()'s encoding
    return {word: (idx + INDEX_FROM) for word, idx in word_index.items()}

model = get_model()
word_index = get_word_index()

def encode_review(text):
    """Convert raw review text into the padded integer sequence the model expects."""
    words = re.findall(r"[a-z']+", text.lower())
    encoded = [word_index.get(w, 2) for w in words]  # 2 = <UNK> token
    encoded = [w if w < VOCAB_SIZE else 2 for w in encoded]
    padded = pad_sequences([encoded], maxlen=MAXLEN)
    return padded

# ---------- Header ----------
st.markdown('<p class="title-text">🎬 Movie Review Sentiment</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Paste a movie review and see what the GRU model thinks</p>', unsafe_allow_html=True)

# ---------- Input ----------
review_text = st.text_area(
    "Your review",
    height=180,
    placeholder="e.g. This movie completely blew me away. The acting was incredible and the story kept me hooked from start to finish...",
    label_visibility="collapsed",
)

predict_btn = st.button("🔮 Analyze Sentiment")

# ---------- Prediction ----------
if predict_btn:
    if not review_text.strip():
        st.warning("Please paste a review first.")
    else:
        encoded_input = encode_review(review_text)
        prediction = model.predict(encoded_input, verbose=0)[0][0]

        is_positive = prediction >= 0.5
        confidence = prediction if is_positive else 1 - prediction

        box_class = "positive-box" if is_positive else "negative-box"
        label = "😊 Positive" if is_positive else "😞 Negative"

        st.markdown(f"""
            <div class="prediction-box {box_class}">
                <p class="prediction-label">{label}</p>
                <p class="confidence-text">Confidence: {confidence*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)

        st.progress(float(prediction))
        st.caption(f"Raw model score: {prediction:.4f} (0.0 = very negative, 1.0 = very positive)")

# ---------- Footer ----------
st.markdown("---")
st.caption("Built with a GRU model trained on IMDB reviews (~87.6% validation accuracy) · TensorFlow + Streamlit")