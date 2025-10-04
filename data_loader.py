import os

from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from  dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None) is not None]
    chunks = []

    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    # Remove empty / whitespace-only strings
    cleaned_texts = [t for t in texts if t and t.strip()]

    if not cleaned_texts:
        raise ValueError("No valid text chunks provided for embedding")

    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=cleaned_texts
    )
    return [item.embedding for item in response.data]
