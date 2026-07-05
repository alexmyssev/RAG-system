import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer #
from typing import Any


class VectorStore:
    def __init__(self):
        self.name = "knowledge_base"
        self.client = PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name = self.name)
        self.embedding_function = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B",  "gpu",  True)

    def add_document(self, documents: list[str], metadatas: list[dict[str, Any]]):
        if not documents:
            return
        embeddings = self.embedding_function(documents).tolist()
        ids = [f"doc_{i}" for i in range(len(documents))]
        self.collection.add(documents = documents, metadatas = metadatas, ids = ids, embeddings = embeddings)

    def query (self, query_texts: list[str], n_results: int) -> dict[str, Any]:
        query_embeddings = self.embedding_function(query_texts).tolist()
        results = self.collection.query(query_embeddings, n_results = n_results)

        return {
            "documents": results["documents"][0] if results["documents"] else None,
            "metadatas": results["metadatas"] if results["metadatas"] else None,
            "ids": results["ids"] if results["ids"] else None,
            "distances": results["distances"] if results["distances"] else None
        }


