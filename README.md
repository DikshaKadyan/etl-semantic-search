# 📑 ETL Pipeline: Semantic Search in Multi-format Documents

A scalable **Extract, Transform, Load (ETL) pipeline** that processes text from **PDF, DOCX, TXT, CSV, and JSON** files, converts them into embeddings, and enables **semantic search** over the documents.  

Built with **Sentence Transformers + NumPy** and deployed with **Streamlit Cloud**.

---

## 🚀 Live Demo
👉 [Click here to try the app](https://etl-semantic-search-wx6jqbm86tjvqyxndzqu53.streamlit.app/)

---

## 🛠️ Features
- Extracts text from multiple file formats:
  - 📄 PDF  
  - 📝 DOCX  
  - 📜 TXT  
  - 📊 CSV  
  - 🔑 JSON  
- Splits text into chunks for semantic representation  
- Generates embeddings with **Hugging Face Sentence Transformers**  
- Stores embeddings in memory (NumPy arrays)  
- Allows **semantic search queries** directly on uploaded files  

---

## ⚙️ Installation (Local Setup)

Clone this repo:
```bash
git clone https://github.com/DikshaKadyan/etl-semantic-search.git
cd etl-semantic-search
