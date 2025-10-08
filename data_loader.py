import os

from llama_index.core import Document
from openai import OpenAI
from llama_index.readers.file import PDFReader, DocxReader
from llama_index.core.node_parser import SentenceSplitter
from  dotenv import load_dotenv
import pandas as pd

load_dotenv()

LOCAL_RAG = os.getenv("LOCAL_RAG", "False").lower() == "true"

if LOCAL_RAG:
    client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"), base_url="http://localhost:11434/v1")
else:
    client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))


splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk(path: str):
    file_extension = file_type(path)
    docs = ""

    if file_extension == "pdf":
        docs = PDFReader().load_data(file=path)
    elif file_extension == "docx" or file_extension == "doc":
        docs = DocxReader().load_data(file=path)
    elif file_extension == "xls" or file_extension == "xlsx":
        docs = excel_to_docs(path)
    else: raise ValueError(f"File extension {file_extension} is not supported.")

    texts = [d.text for d in docs if getattr(d, "text", None) is not None]
    chunks = []

    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    # Remove empty / whitespace-only strings - Avoiding useless or blank data to the embedding model
    cleaned_texts = [t for t in texts if t and t.strip()]

    if not cleaned_texts:
        raise ValueError("No valid text chunks provided for embedding")

    response = client.embeddings.create(
        model=os.getenv("EMBED_MODEL"),
        input=cleaned_texts
    )
    # response.data:
    # [
    #     {"embedding": [0.0123, -0.0456, 0.9876, ...]},
    #     {"embedding": [0.0021, -0.0123, 0.6543, ...]},
    #      ...
    # ]

    return [item.embedding for item in response.data]
    # return
    # [
    #   [0.0123, -0.0456, 0.9876, ...],
    #   [0.0021, -0.0123, 0.6543, ...]
    # ]

def file_type(file_path: str):
    file_extension = file_path.split(".")[-1].lower()
    return file_extension


def excel_to_docs(path: str):
    sheets = pd.read_excel(path, sheet_name=None)
    docs = []

    for sheet_name, df in sheets.items():
        # Convert all cells to string, skip NaN
        rows_text = []
        for _, row in df.iterrows():
            row_values = [str(v) for v in row if pd.notna(v)]
            if row_values:  # skip empty rows
                rows_text.append(" ".join(row_values))

        if rows_text:
            text = f"Sheet: {sheet_name}\n" + "\n".join(rows_text)
            docs.append(Document(text=text, metadata={"file_name": os.path.basename(path), "sheet_name": sheet_name}))

    return docs