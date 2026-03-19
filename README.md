# AI Legislative Analyzer 🚀

## 📌 Overview
AI Legislative Analyzer is a web-based NLP application that processes PDF documents, extracts text, and performs:

- Text summarization using transformer models (BART)
- Token compression analysis
- Keyword extraction using NLP techniques
- Chunk-wise processing for handling large documents

The project demonstrates real-world techniques like context pruning, token optimization, and scalable summarization pipelines.

---

## 🎯 Problem Statement
Large documents such as legal reports, research papers, and policy documents are difficult to read and analyze manually. This project aims to:

- Reduce document size while preserving meaning
- Provide concise summaries
- Extract important keywords
- Measure token reduction efficiency

---

## ⚙️ Features

- 📄 Upload PDF documents
- 🧠 AI-based summarization using `facebook/bart-large-cnn`
- ✂️ Token compression with chunking strategy
- 🔑 Keyword extraction using NLP + stopword removal
- 📊 Token statistics (original vs compressed)
- 📉 Token reduction percentage
- 🌐 Simple web UI using Flask + Bootstrap

---

## 🏗️ Architecture
PDF Upload → Text Extraction → Cleaning → Chunking →
BART Summarization → Merge Summary →
Keyword Extraction → Token Analysis → UI Display

---

## 🧪 Techniques Used

- Context Pruning (sentence tokenization + chunking)
- Token Compression (transformer summarization)
- NLP preprocessing (regex cleaning, stopword removal)
- Frequency-based keyword extraction
- Pipeline-based inference using HuggingFace Transformers

---

## 📊 Measurable Outputs

- Original token count
- Compressed token count
- Token reduction percentage
- Generated summary
- Top keywords

---

## 🛠️ Tech Stack

- Python
- Flask
- HuggingFace Transformers
- BART Model (`facebook/bart-large-cnn`)
- NLTK
- PyPDF2
- HTML, Bootstrap

---

## 🚀 Installation & Setup

### 1. Clone the repository