
from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import PyPDF2
import re

# 🔹 Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# 🔹 Load BART model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# 🔹 Home Page
@app.route('/')
def home():
    return render_template('index.html')


# 🔹 Extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + " "
    return text


# 🔹 Light Text Preprocessing (FOR BART)
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9.,\s]', '', text)  # keep sentence structure
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# 🔹 Token count
def count_tokens(text):
    return len(text.split())


# 🔹 Compress text using BART
def compress_text(text):
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len((current_chunk + sentence).split()) <= 400:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    summary = ""

    for chunk in chunks:
        result = summarizer(
            chunk,
            max_length=130,
            min_length=50,
            do_sample=False
        )
        summary += result[0]['summary_text'] + " "

    return summary.strip()


# 🔹 Keyword Extraction (WITH STOPWORD REMOVAL ✅)
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))

    words = nltk.word_tokenize(text.lower())

    # 🔥 remove stopwords like is, of, by, like
    filtered_words = [
        w for w in words
        if w.isalnum() and w not in stop_words and len(w) > 3
    ]

    freq = {}
    for w in filtered_words:
        freq[w] = freq.get(w, 0) + 1

    return sorted(freq, key=freq.get, reverse=True)[:10]


# 🔥 Upload PDF API
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']

    # Step 1: Extract text
    text = extract_text_from_pdf(file)

    if not text:
        return jsonify({"error": "No text found in PDF"})

    # Step 2: Clean text (ONLY light preprocessing)
    cleaned_text = clean_text(text)

    # Step 3: Token count
    original_tokens = count_tokens(cleaned_text)

    # Step 4: BART summarization (NO stopword removal here ❗)
    compressed = compress_text(cleaned_text)
    compressed_tokens = count_tokens(compressed)

    # Step 5: Reduction %
    reduction = ((original_tokens - compressed_tokens) / original_tokens) * 100

    # Step 6: Keyword extraction (WITH stopword removal)
    keywords = extract_keywords(cleaned_text)

    return jsonify({
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "token_reduction_percent": round(reduction, 2),
        "summary": compressed,
        "keywords": keywords
    })


# 🔹 Run App
if __name__ == '__main__':
    app.run(debug=True)
