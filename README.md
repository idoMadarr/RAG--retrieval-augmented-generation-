# 🧠 RAG System with FastAPI, Qdrant, Ollama, and Streamlit

A lightweight Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **Ollama/OpenAI**, **Qdrant** (vector database), and **Streamlit** for an interactive UI.  
Supports **PDF**, **Word (DOCX)**, and **Excel** file ingestion and dynamic switching between **Ollama** (local models) and **OpenAI**.

---

## 🚀 Features

- 🔄 Dynamic switch between **Ollama** and **OpenAI**
- 🧩 Vector storage using **Qdrant** (via Docker)
- 📄 Multi-format document ingestion (PDF, DOCX, XLSX)
- 💬 Question answering over your private documents
- 🎨 Simple and clean **Streamlit** UI
- ⚡ Powered by **FastAPI** backend

---

## ⚙️ Environment Setup

Create a `.env` file (example):

```env
OPEN_AI_KEY="YOUR_API_KEY"
SERVER="http://127.0.0.1:8000"

# OpenAI Credentials
#LLM=gpt-4o-mini
#EMBED_MODEL=text-embedding-3-large
#LOCAL_RAG=False
#VECTOR_DIMENSION=3072

# Ollama Credentials
LLM=gemma3:12b
EMBED_MODEL=mxbai-embed-large
LOCAL_RAG=True
VECTOR_DIMENSION=1024
```

## 🚀 Running the Project
### * (Recommend) Run Full Project Script
```
python scripts.py
```

### 1. Activate Virtual Environment (optional)
```
.venv\Scripts\Activate.ps1
```

### 2. Make sure Ollama is running

### 3. Start Qdrant (Vector Database)
```
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

### 4. Start FastAPI Backend
```
uv run uvicorn main:app
```

### 5. Start Streamlit UI
```
streamlit run interface.py
```