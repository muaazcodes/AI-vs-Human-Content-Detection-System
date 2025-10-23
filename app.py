import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
from textstat import flesch_reading_ease

# --- Page Config ---
st.set_page_config(page_title="AI vs Human Detector", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    /* Poora app ka background aur default text */
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }

    /* Label text color fix (for 'Enter your text here...' heading) */
    label, .stTextArea label {
        color: #FFFFFF !important;
        font-weight: bold;
    }

    /* Textarea style (user ka input box) */
    textarea {
        background-color: #111111 !important;
        color: #FFFFFF !important;      /* likhne wala text white */
        border: 1px solid #00FF00 !important;
        border-radius: 10px !important;
    }

    /* Placeholder color (Enter your text here...) */
    textarea::placeholder {
        color: #BBBBBB !important;       /* halka gray */
    }

    /* Buttons ka design */
    div.stButton > button {
        background-color: #00FF00;
        color: #000000;
        border: none;
        padding: 0.6rem 1.2rem;
        font-size: 16px;
        font-weight: bold;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #00cc00;
        color: #000000;
        transform: scale(1.05);
    }

    /* DataFrame table (debugging section) */
    .stDataFrame div[data-testid="stVerticalBlock"] {
        background-color: #111111;
        color: #FFFFFF;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #00FF00;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# --- Helper Functions ---
def load_pipeline():
    try:
        with open("Pipeline.pkl", "rb") as f:
            pipeline = pickle.load(f)
        return pipeline
    except FileNotFoundError:
        st.error("⚠️ Pipeline.pkl file nahi mili! File ko isi folder me rakho.")
        return None

def extract_features(text):
    words = re.findall(r'\b\w+\b', text.lower())
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]

    word_count = len(words)
    char_count = len(text)
    sentence_count = len(sentences)
    avg_word_length = char_count / word_count if word_count > 0 else 0
    lexical_diversity = len(set(words)) / word_count if word_count > 0 else 0
    punctuation_ratio = len(re.findall(r'[.,;!?]', text)) / char_count if char_count > 0 else 0
    sentence_lengths = [len(s.split()) for s in sentences]
    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
    burstiness = np.std(sentence_lengths) / avg_sentence_length if avg_sentence_length > 0 else 0

    try:
        fre = flesch_reading_ease(text)
    except:
        fre = 0.0

    grammar_errors = 0.0
    gunning_fog_index = 0.0
    passive_voice_ratio = 0.0
    predictability_score = 0.0

    data = {
        'word_count': [word_count],
        'character_count': [char_count],
        'sentence_count': [sentence_count],
        'lexical_diversity': [lexical_diversity],
        'avg_sentence_length': [avg_sentence_length],
        'avg_word_length': [avg_word_length],
        'punctuation_ratio': [punctuation_ratio],
        'burstiness': [burstiness],
        'flesch_reading_ease': [fre],
        'grammar_errors': [grammar_errors],
        'gunning_fog_index': [gunning_fog_index],
        'passive_voice_ratio': [passive_voice_ratio],
        'predictability_score': [predictability_score]
    }

    return pd.DataFrame(data)

# --- Main App ---
st.markdown("<h1>🤖 AI vs Human Detector</h1>", unsafe_allow_html=True)
st.write("Paste your text below — the app will detect whether it’s Human-written or AI-generated.")

st.markdown("---")

pipeline = load_pipeline()

if pipeline:
    text_input = st.text_area(
        "Enter your text here (minimum 20 words recommended):",
        height=250,
        placeholder="Type or paste your content..."
    )

    if st.button("Detect"):
        if len(text_input.split()) < 20:
            st.warning("⚠️ Minumum 20 words only.")
        else:
            with st.spinner("🧠 Analyzing linguistic features..."):
                features_df = extract_features(text_input)

                # Ensure all features match pipeline
                missing_cols = [col for col in pipeline.feature_names_in_ if col not in features_df.columns]
                for col in missing_cols:
                    features_df[col] = 0.0
                features_df = features_df[pipeline.feature_names_in_]

                try:
                    prediction = pipeline.predict(features_df)
                except ValueError as e:
                    st.error(f"Prediction failed! {e}")
                    st.stop()

                st.subheader("🔍 Detection Result")
                if prediction[0] == 1:
                    st.success("👨 Human-Written Content")
                else:
                    st.error("🤖 AI-Generated Content")

                st.markdown("---")
                st.write("🧩 Extracted Features (Debugging):")
                st.dataframe(features_df)
