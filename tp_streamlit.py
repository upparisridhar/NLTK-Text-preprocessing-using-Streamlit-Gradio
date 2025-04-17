import streamlit as st
import re
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

def text_pre(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filt_tokens = [token for token in tokens if token not in stop_words]

    pos_tags = pos_tag(filt_tokens)
    
    lemma = WordNetLemmatizer()
    lemma_tokens = [lemma.lemmatize(token) for token, tag in pos_tags]

    stemmer = PorterStemmer()
    stem_tokens = [stemmer.stem(token) for token in filt_tokens]

    return {
        "original": tokens,
        "filtered": filt_tokens,
        "lemmatized": lemma_tokens,
        "stemmed": stem_tokens,
        "pos_tags": pos_tags
    }

st.title("Text Preprocessing App")

user_input = st.text_area("Enter your sentence:")

if user_input:
    result = text_pre(user_input)
    
    st.subheader("🧾 Output (JSON format)")
    st.json(result)

    # Convert result to JSON string for download
    json_str = json.dumps(result, indent=4)
    st.download_button(
        label="📥 Download Result as JSON",
        data=json_str,
        file_name="text_preprocessing_result.json",
        mime="application/json"
    )
