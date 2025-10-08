import streamlit as st
import requests
import re


API_URL = "http://127.0.0.1:8000"  # your FastAPI server

st.set_page_config(page_title="RAG System", layout="wide")

# --- Custom CSS for RTL ---
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

st.title("📚 RAG System")
st.text("Upload multiple PDFs and ask questions based on their content.")

# --- Initialize session state safely ---
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# --- Upload section ---
st.header("📤 Upload PDF or Doc file")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "doc", "docx"])

if st.button("Upload"):
    if uploaded_file:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Uploading..."):
            response = requests.post(
                f"{API_URL}/upload_file",
                json={"file_path": temp_path, "source_id": uploaded_file.name}
            )

        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ File uploaded and indexed! Chunks ingested: {result.get('ingested', '?')}")
            if uploaded_file.name not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(uploaded_file.name)
        else:
            st.error(f"Error uploading: {response.text}")
    else:
        st.warning("Please choose a file first.")

# --- Ask a Question section ---
st.header("💬 Ask a Question")

if st.session_state.uploaded_files:
    selected_file = st.selectbox("Select a file to query:", st.session_state.uploaded_files)
    question = st.text_input("Ask me anything about your selected file:")
    top_k = st.number_input("Top K results", value=5, min_value=1, max_value=20, step=1)

    def is_hebrew(text):
        return bool(re.search(r'[\u0590-\u05FF]', text))

    if st.button("Send"):
        if not question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner(text="Fetching results..."):
                response = requests.post(
                    f"{API_URL}/query-pdf",
                    json={"question": question, "top_k": top_k, "source_id": selected_file}
                )


            if response.status_code == 200:
                data = response.json()
                st.subheader("🧠 Answer")
                answer_class = "rtl" if is_hebrew(data["answer"]) else "ltr"
                st.markdown(f"<div class='{answer_class}'>{data['answer']}</div>", unsafe_allow_html=True)
                st.subheader("📄 Sources")
                for src in data.get("sources", []):
                    src_class = "rtl" if is_hebrew(src) else "ltr"
                    st.markdown(f"<div class='{src_class}'>- {src}</div>", unsafe_allow_html=True)
            else:
                st.error(f"Error: {response.text}")
else:
    st.warning("Please upload at least one PDF to start asking questions.")
