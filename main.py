import logging
from fastapi import FastAPI
from dotenv import load_dotenv

import inngest
import inngest.fast_api
from inngest.fast_api import serve
from inngest.experimental import ai

from data_loader import load_and_chunk, embed_texts
from rag_types import RAGQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc

import uuid
import os
import datetime

from vector_db import QdrantStorage

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(context: inngest.Context):
    # Step 1 - load pdf data
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)


    # Step 2 - embed and store
    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        # Filter out empty chunks
        filtered_chunks = [c for c in chunks if c.strip() != ""]

        vectors = embed_texts(filtered_chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(filtered_chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(filtered_chunks))]

        QdrantStorage().upsert(ids, vectors, payloads)
        return RAGUpsertResult(ingested=len(filtered_chunks))

    chunks_and_src = await context.step.run("load-and-chunk", lambda: _load(context), output_type=RAGChunkAndSrc)
    ingested = await context.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf")
)
async def rag_query_pdf(context: inngest.Context):
    def _search(question: str, top_k: int = 5)  -> RAGSearchResult:
        query_vector = embed_texts([question])[0]
        store = QdrantStorage()
        result = store.search(query_vector, top_k)
        return RAGSearchResult(contexts=result["contexts"], sources=result["sources"])

    question = context.event.data["question"]
    top_k = int(context.event.data.get("top_k", 5))

    found = await context.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the following questions:\n\n"
        f"Context: {context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPEN_AI_KEY"),
        model="gpt-4o-mini"
    )

    response = await  context.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                { "role": "system", "content": "You answer questions using only the provided context" },
                { "role": "user", "content": user_content }
            ]
        }
    )

    answer = response["choices"][0]["message"]["content"].strip()
    return { "answer": answer, "sources": found.sources, "num_contexts": len(found.contexts) }


app = FastAPI()

# # Simple GET endpoint for testing
# @app.get("/")
# def read_root():
#     return {"message": "Hello, FastAPI is working!"}
#
# # Another test endpoint with parameter
# @app.get("/hello/{name}")
# def say_hello(name: str):
#     return {"message": f"Hello, {name}!"}

serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf])
