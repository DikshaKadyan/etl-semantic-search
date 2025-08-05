import streamlit as st
import fitz  # PDF
import docx
import pandas as pd
import json
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ---------- Initialize ----------
chroma_client = chromadb.Client()
collection = None  # will be created when file is uploaded

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


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


# ---------- Load ----------
def load_data(chunks_with_embeddings):
    global collection
    # delete old collection and recreate fresh
    try:
        chroma_client.delete_collection("docs")
    except:
        pass
    collection = chroma_client.create_collection("docs")

    for idx, (chunk, emb) in enumerate(chunks_with_embeddings):
        collection.add(
            documents=[chunk],
            embeddings=[emb],
            ids=[str(idx)]
        )


# ---------- Streamlit UI ----------
st.title("📑 ETL Pipeline: Semantic Search in Multi-format Documents")
st.write("Upload a file (PDF, DOCX, TXT, CSV, JSON) and ask questions about it.")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt", "csv", "json"])

if uploaded_file:
    text = extract_text(uploaded_file)
    st.success("✅ File uploaded and text extracted")

    chunks_with_embeddings = transform_text(text)
    load_data(chunks_with_embeddings)
    st.success("✅ Text processed and stored in fresh vector DB")

    query = st.text_input("Ask a question about the document:")
    if query and collection:
        results = collection.query(query_texts=[query], n_results=3)
        st.write("### 🔎 Top Results")
        for doc in results['documents'][0]:
            st.write("-", doc)
