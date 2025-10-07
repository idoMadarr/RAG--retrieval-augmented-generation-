import streamlit as st
import requests
import re

API_URL = "http://127.0.0.1:8000"  # your FastAPI server

st.set_page_config(page_title="RAG System", layout="wide")

# Custom CSS for RTL
st.markdown("""
    <style>
    .rtl {
        direction: rtl;
        text-align: right;
        font-family: "Arial", sans-serif;
    }
    .ltr {
        direction: ltr;
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

st.title("RAG System")
st.text("RAG (Retrieval-Augmented Generation) is an AI framework that combines the strengths of traditional information retrieval systems (such as search and databases) with the capabilities of generative large language models (LLMs).")


# --- Upload PDF ---
st.header("Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if st.button("Upload PDF"):
    if uploaded_file is not None:
        # Save temporarily to disk
        with open(f"temp_{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Send to FastAPI
        response = requests.post(
            f"{API_URL}/upload_pdf",
            json={"pdf_path": f"temp_{uploaded_file.name}"}
        )
        if response.status_code == 200:
            st.success(f"PDF uploaded and indexed! Chunks ingested: {response.json()['ingested']}")
        else:
            st.error(f"Error: {response.text}")

# --- Ask a Question ---
st.header("Ask a Question")
question = st.text_input("Your me anything about your file...")
top_k = st.number_input("Top K results", value=5, min_value=1, max_value=20, step=1)

def is_hebrew(text):
    return bool(re.search(r'[\u0590-\u05FF]', text))

if st.button("Ask"):
    if question:
        response = requests.post(
            f"{API_URL}/query-pdf",
            json={"question": question, "top_k": top_k}
        )
        if response.status_code == 200:
            data = response.json()
            st.subheader("Answer")

            answer_class = "rtl" if is_hebrew(data["answer"]) else "ltr"
            st.markdown(f"<div class='{answer_class}'>{data['answer']}</div>", unsafe_allow_html=True)

            st.subheader("Sources")
            for src in data["sources"]:
                src_class = "rtl" if is_hebrew(src) else "ltr"
                st.markdown(f"<div class='{src_class}'>- {src}</div>", unsafe_allow_html=True)
        else:
            st.error(f"Error: {response.text}")