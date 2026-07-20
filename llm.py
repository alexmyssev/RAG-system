import ollama#
import time
from config import AGENT_CONFIG, PROMPT_FORMAT, OLLAMA_MODEL, OLLAMA_HOST
from vector_storage import VectorStore


class Model:
    def __init__(self, vector_store: VectorStore, model_name: str = OLLAMA_MODEL, config: dict[str, any] = None, host: str = OLLAMA_HOST):

        self.model_name = model_name
        self.config = config or AGENT_CONFIG.copy()
        if host:
          self.client = ollama.Client(host=host)
        else:
            self.client =  ollama.Client()
        try:
            self.client.show(self.model_name)
        except Exception:
            raise RuntimeError(
                f"Model '{self.model_name}' not found. "
                f"Run: ollama pull {self.model_name}"
            )
        self.vector_store = vector_store
        self.n_result = 5

    def _generation_options(self) -> dict[str, any]:
        return {
            "temperature": self.config.get("temperature", 0.7),
            "num_predict": self.config.get("max_output_tokens", 2048),
            "top_p": self.config.get("top_p", 0.8),
            "top_k": self.config.get("top_k", 40),
        }

    def form_prompt(self, query: str) -> dict[str, any]:
        n_results = self.n_result
        results = self.vector_store.retriever([query], n_results)
        retrieved_documents = results["documents"]
        metadatas = results["metadatas"]#

        context = " ".join(retrieved_documents)
        system_instruction = "маркер: 12A3B; Ты - ассистент, отвечающий на вопросы на основе данных, извлеченных из переданного контекста; выводи ответ на русском языке; при  использовании информации из контекста, указывай, что информация взята из Википедии; маркер: B3A21"

        prompt = PROMPT_FORMAT.format(system_instruction = system_instruction, context = context, question = query)

        return{
            "prompt": prompt,
            "metadatas": metadatas,
            "retrieved_documents": retrieved_documents,
            "distances": results["distances"]
        }


    def generate(self, query: str, **kwargs) -> str:
        start_time = time.time()
        prompt = self.form_prompt(query)
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt["prompt"],
                options=self._generation_options(),
                **kwargs,
            )

            text = response.get("response")

            if not text:
                raise Exception("Empty response from model")

            time_spent = time.time() - start_time


            for i in range(len(prompt["retrieved_documents"])):
                document = prompt["retrieved_documents"][i]
                metadatas = prompt["metadatas"][i]
                distance = prompt["distances"][i]
                print(f" --- №{i} --- ")
                print(f"Source: {metadatas["source"]}")
                print(f"Distance: {distance:.6}")
                print(f"Text: {document}")

            print(f"Time spent: {time_spent:.6} seconds")

            return text

        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")








