import os

from openai import OpenAI
from llama_index.readers.file import PDFReader, DocxReader
from llama_index.core.node_parser import SentenceSplitter
from  dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk(path: str):
    file_extension = file_type(path)
    docs = ""

    if file_extension == "pdf":
        docs = PDFReader().load_data(file=path)
    elif file_extension == "docx" or file_extension == "doc":
        docs = DocxReader().load_data(file=path)
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
        model=EMBED_MODEL,
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