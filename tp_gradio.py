import gradio as gr
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Function to map POS tags to WordNet format
def get_wordnet_pos(tag):
    if tag.startswith('VB'):
        return wordnet.VERB
    elif tag.startswith('JJ'):
        return wordnet.ADJ
    elif tag.startswith('RB'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered = [t for t in tokens if t not in stop_words]

    pos_tags = pos_tag(filtered)
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]

    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in filtered]

    output = f"""
    🔹 Original Tokens: {tokens}
    🔹 Filtered (No Stopwords): {filtered}
    🔹 POS Tags: {pos_tags}
    🔹 Lemmatized: {lemmatized}
    🔹 Stemmed: {stemmed}
    """
    return output.strip()

# Gradio interface
demo = gr.Interface(
    fn=preprocess_text,
    inputs="text",
    outputs="text",
    title="🧹 Text Preprocessing with Gradio",
    description="Enter a sentence and see tokenization, stopword removal, POS tagging, lemmatization, and stemming."
)

if __name__ == "__main__":
    demo.launch(share=True)
