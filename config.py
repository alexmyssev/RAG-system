AGENT_CONFIG = {
    "temperature": 0.7,
    "max_output_tokens": 2048,
    "top_p": 0.8,
    "top_k": 40,
}

PROMPT_FORMAT = """ 
    System_instruction: {system_instruction}
    Context: {context} 
    Question: {question} 
    """

OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_HOST = "http://localhost:11434"
