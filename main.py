import logging
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from openai import OpenAI

from data_loader import load_and_chunk, embed_texts
from rag_types import RAGQueryResult, RAGSearchResult, RAGUpsertResult

import uuid
import os

from vector_db import QdrantStorage

load_dotenv()
app = FastAPI()

logger = logging.getLogger("uvicorn")
client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

@app.post('/upload_pdf')
async def upload_pdf(body: dict):
    pdf_path = body.get("pdf_path", None)
    source_id = body.get("source_id", None)

    if not pdf_path or not source_id:
        raise HTTPException(status_code=400, detail=f"Invalid request body")

    # Load & Chunk
    logger.info(f"Loading and Chunking PDF: {pdf_path}")
    chunks = load_and_chunk(pdf_path)
    chunks = [c for c in chunks if c.strip()]
    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks extracted from PDF.")

    # Embed & Upsert
    vectors = embed_texts(chunks)

    # Creating ID per chunk → required for the database
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
    # [ "f47ac10b-58cc-4372-a567-0e02b2c3d479", "a4c9b8f7-3e45-4f5c-91ef-79b6b9f7c77f", ... ]

    payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
    # [ {"source": "my-pdf", "text": "Chunk one text"}, {"source": "my-pdf", "text": "Chunk two text"}, ... ]

    QdrantStorage().upsert(ids, vectors, payloads)

    result = RAGUpsertResult(ingested=len(chunks))
    return result.model_dump()


@app.post("/query-pdf")
async def query_pdf(body: dict):
    question = body.get("question")
    top_k = int(body.get("top_k", 5))

    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question' in request body.")

    logger.info(f"Embedding and searching for question: {question}")

    # Step 1 — embed and search
    query_vector = embed_texts([question])[0]
    # after we embed the question, we take the first element in the embed data (for longer questions it can be a bit problematic, but it depends on the embedding model)

    result = QdrantStorage().search(query_vector, top_k)
    found = RAGSearchResult(contexts=result["contexts"], sources=result["sources"])

    if not found.contexts:
        return {"answer": "No relevant context found.", "sources": [], "num_contexts": 0}

    # Step 2 — create LLM prompt
    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_prompt = (
        "Use the following context to answer the question:\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely using only the context above."
    )

    logger.info("Calling OpenAI GPT model for answer generation...")

    # Step 3 — generate answer
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You answer questions using only the provided context."},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()
    result = RAGQueryResult(answer=answer, sources=found.sources, num_contexts=len(found.contexts))

    return result.model_dump() # model_dump() Convert pydantic object into a plain python dictionary

