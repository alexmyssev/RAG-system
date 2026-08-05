from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from config import DEVICE


class VectorStore:
    def __init__(self):
        self.name = "knowledge_base"
        self.client = PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name = self.name)
        self.embedding_function = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B",  device=DEVICE, trust_remote_code=True)



    def add_document(self, documents: list[str], metadatas: list[dict[str, any]]):
        if not documents:
            return
        embeddings = self.embedding_function.encode(documents, batch_size=16, show_progress_bar=True).tolist()
        ids = [f"doc_{i}" for i in range(len(documents))]
        max_batch_size = 5000
        for i in range(0, len(documents), max_batch_size):
            batch_end = i + max_batch_size
            self.collection.add(documents=documents[i:batch_end], metadatas=metadatas[i:batch_end], ids=ids[i:batch_end], embeddings=embeddings[i:batch_end])
            print(f"Добавлен батч {i}–{min(batch_end, len(documents))} из {len(documents)}")

    def retriever (self, query_texts: str, n_results: int, full_access: bool) -> dict[str, any]:
        query_embeddings = self.embedding_function.encode(query_texts).tolist()

        lim_access = None
        if not full_access:
            lim_access = {"status": "not confidential"}

        results = self.collection.query(query_embeddings, n_results=n_results, where=lim_access)
        return {
            "documents": results["documents"][0] if results["documents"] else None,
            "metadatas": results["metadatas"][0] if results["metadatas"] else None,
            "ids": results["ids"][0] if results["ids"] else None,
            "distances": results["distances"][0] if results["distances"] else None
        }



