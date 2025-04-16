import gradio as gr
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered = [t for t in tokens if t not in stop_words]

    pos_tags = pos_tag(filtered)
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in filtered]

    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in filtered]

    output = f"""
    🔹 Original Tokens: {tokens}
    🔹 Filtered: {filtered}
    🔹 Lemmatized: {lemmatized}
    🔹 Stemmed: {stemmed}
    """
    return output

demo = gr.Interface(fn=preprocess_text, inputs="text", outputs="text",
                    title="🧹 Text Preprocessing with Gradio",
                    description="Enter a sentence and see tokenization, stopword removal, lemmatization, and stemming.")

if __name__ == "__main__":
    demo.launch(share=True)