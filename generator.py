from vector_storage import VectorStore
from config import AGENT_CONFIG, PROMPT_FORMAT
from llm import Model

class Generator(Model):
    def __init__(self, vector_store: VectorStore):
        Model.__init__(self,config=AGENT_CONFIG)
        self.vector_store = vector_store
        self.n_result = 5

    def generate_response(self, query: str) -> dict[str, any]:
        n_results = self.n_result
        results = self.vector_store.query(query, n_results)
        retrieved_documents = results["documents"]
        metadatas = results["metadatas"]
        context = " ".join(retrieved_documents)

        prompt = PROMPT_FORMAT(context = context, question = query)

        answer = self.generate_response(prompt)

        return {
            "answer": answer,
            "context": context,
            "retrieved_chunks": retrieved_documents,
            "metadatas": metadatas
        }

