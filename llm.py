import ollama#
from config import AGENT_CONFIG, OLLAMA_HOST, OLLAMA_MODEL

class Model:
    def __init__(self, model_name: str = OLLAMA_MODEL, config: dict[str, any] = None, host: str = OLLAMA_HOST):

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

    def _generation_options(self) -> dict[str, any]:
        return {
            "temperature": self.config.get("temperature", 0.7),
            "num_predict": self.config.get("max_output_tokens", 2048),
            "top_p": self.config.get("top_p", 0.8),
            "top_k": self.config.get("top_k", 40),
        }

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options=self._generation_options(),
                **kwargs,
            )

            text = response.get("response")

            if not text:
                raise Exception("Empty response from model")
            return text.strip()
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")


