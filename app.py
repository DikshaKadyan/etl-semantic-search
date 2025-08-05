import streamlit as st
import fitz  # PDF
import docx
import pandas as pd
import json
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ---------- Initialize ----------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

docs = []
embeddings = None


# ---------- Extract ----------
def extract_text(file):
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return " ".join([page.get_text() for page in doc])
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return " ".join([para.text for para in doc.paragraphs])
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".csv"):
        df = pd.read_csv(file)
        return df.to_string()
    elif file.name.endswith(".json"):
        data = json.load(file)
        return json.dumps(data, indent=2)
    else:
        return ""


# ---------- Transform ----------
def transform_text(text):
    chunks = splitter.split_text(text)
    vectors = embedder.encode(chunks, show_progress_bar=False)
    return chunks, np.array(vectors, dtype="float32")


# ---------- Query ----------
def query_data(query, top_k=3):
    global docs, embeddings
    if embeddings is None:
        return []
    query_vec = embedder.encode([query]).astype("float32")
    # Cosine similarity
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    sims = np.dot(embeddings, query_vec.T).flatten() / norms
    top_idx = sims.argsort()[-top_k:][::-1]
    return [docs[i] for i in top_idx]


# ---------- Streamlit UI ----------
st.title("📑 ETL Pipeline: Semantic Search in Multi-format Documents")
st.write("Upload a file (PDF, DOCX, TXT, CSV, JSON) and ask questions about it.")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt", "csv", "json"])

if uploaded_file:
    text = extract_text(uploaded_file)
    st.success("✅ File uploaded and text extracted")

    docs, embeddings = transform_text(text)
    st.success("✅ Text processed and stored in memory")

    query = st.text_input("Ask a question about the document:")
    if query and embeddings is not None:
        results = query_data(query, top_k=3)
        st.write("### 🔎 Top Results")
        for doc in results:
            st.write("-", doc)
