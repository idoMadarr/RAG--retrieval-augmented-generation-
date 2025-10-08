from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, MatchValue
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter

class QdrantStorage:
    def __init__(self, url="http://localhost:6333/", collection="docs", dim=3072):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(collection_name=self.collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))


    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)


    def search(self, query_vector, top_k = 5, source_id=None):
        query_filter = None

        if source_id:
            query_filter = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_id))])

        results = self.client.search(self.collection, query_vector=query_vector, limit=top_k, query_filter=query_filter)
        # results = [
        #     ScoredPoint(id='d80e027c-b8b7-586c-bb86-8fe8054422a5', version=2, score=0.18, payload={'text': 'chunk1', 'source': 'file.pdf'}),
        #     ScoredPoint(id='b561fc5e-a439-531a-b11e-3ddf2b9935e3', version=2, score=0.22, payload={'text': 'chunk2', 'source': 'file.pdf'}),
        # ]

        contexts = []
        sources = set()

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")      # Chunk content
            source = payload.get("source", "")  # File name
            if text:
                contexts.append(text)
                sources.add(source)

        return { "contexts": contexts, "sources": list(sources) }
        # contexts = [
        #     "This is chunk 2",
        #     "This is chunk 1"
        # ]
        #
        # sources = {"example.pdf"}