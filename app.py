import streamlit as st
import fitz  # PDF
import docx
import pandas as pd
import json
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ---------- Initialize ----------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

index = None
docs = []


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
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    return list(zip(chunks, embeddings))


# ---------- Load into FAISS ----------
def load_data(chunks_with_embeddings):
    global index, docs
    docs = [chunk for chunk, _ in chunks_with_embeddings]
    vectors = np.array([emb for _, emb in chunks_with_embeddings]).astype("float32")
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 distance index
    index.add(vectors)


# ---------- Query ----------
def query_data(query, top_k=3):
    query_vector = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, top_k)
    return [docs[i] for i in indices[0]]


# ---------- Streamlit UI ----------
st.title("📑 ETL Pipeline: Semantic Search in Multi-format Documents")
st.write("Upload a file (PDF, DOCX, TXT, CSV, JSON) and ask questions about it.")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt", "csv", "json"])

if uploaded_file:
    text = extract_text(uploaded_file)
    st.success("✅ File uploaded and text extracted")

    chunks_with_embeddings = transform_text(text)
    load_data(chunks_with_embeddings)
    st.success("✅ Text processed and stored in FAISS index")

    query = st.text_input("Ask a question about the document:")
    if query and index is not None:
        results = query_data(query, top_k=3)
        st.write("### 🔎 Top Results")
        for doc in results:
            st.write("-", doc)
