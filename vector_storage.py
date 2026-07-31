from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer #
from transformers import AutoModel
from config import DEVICE
#from instructions_detector import Detector

class VectorStore:
    def __init__(self):
        self.name = "knowledge_base"
        self.client = PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name = self.name)
        self.embedding_function = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B",  device=DEVICE, trust_remote_code=True)

        self.reranker_model = AutoModel.from_pretrained("jinaai/jina-reranker-v3", dtype="auto", trust_remote_code=True)
        self.reranker_model.to(DEVICE)
        self.reranker_model.eval()

        #self.instructions_detector = Detector()

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

    def retriever (self, query_text: str, n_results: int, full_access: bool) -> dict[str, any]:
        query_embeddings = self.embedding_function.encode(query_text).tolist()

        lim_access = None
        if not full_access:
            lim_access = {"status": "not confidential"}

        results = self.collection.query(query_embeddings, n_results = n_results, where = lim_access)
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []
        if not documents:
            return {"documents": [], "metadatas": [], "distances": [], "rerank_scores": []}
    
        return self.to_rerank(query_text, documents, metadatas, distances)

    def to_rerank (self, query_text: str, documents: list[str], metadatas: list[dict], distances: list[float]):

        top_k = 5
        result = self.reranker_model.rerank(query_text, documents)
        result = result[:top_k]

        reranked_documents = []
        reranked_metadatas = []
        reranked_distances = []
        reranked_scores = []

        for res in result:
            #if self.instructions_detector.is_malicious(res["document"]):
            #    continue
            retriever_index = res["index"]
            reranked_documents.append(res["document"])
            reranked_metadatas.append(metadatas[retriever_index])
            reranked_distances.append(distances[retriever_index])
            reranked_scores.append(res["relevance_score"])

        return {
            "documents": reranked_documents,
            "metadatas": reranked_metadatas,
            "distances": reranked_distances,
            "rerank_scores": reranked_scores
        }


