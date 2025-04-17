import streamlit as st
import re
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Function to get correct POS for lemmatization
def get_wordnet_pos(tag):
    if tag.startswith('VB'):
        return wordnet.VERB
    elif tag.startswith('JJ'):
        return wordnet.ADJ
    elif tag.startswith('RB'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Text preprocessing function
def text_pre(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filt_tokens = [token for token in tokens if token not in stop_words]

    # POS tagging for correct lemmatization
    pos_tags = pos_tag(filt_tokens)
    
    # Lemmatization using correct POS
    lemma = WordNetLemmatizer()
    lemma_tokens = [lemma.lemmatize(token, get_wordnet_pos(tag)) for token, tag in pos_tags]

    # Stemming using PorterStemmer
    stemmer = PorterStemmer()
    stem_tokens = [stemmer.stem(token) for token in filt_tokens]

    return {
        "original": tokens,
        "filtered": filt_tokens,
        "lemmatized": lemma_tokens,
        "stemmed": stem_tokens,
        "pos_tags": pos_tags
    }

# Streamlit app
st.title("Text Preprocessing App")

# Text input from user
user_input = st.text_area("Enter your sentence:")

if user_input:
    result = text_pre(user_input)
    
    # Display result in JSON format
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