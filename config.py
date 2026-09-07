import os

AGENT_CONFIG = {
    "temperature": 0.7,
    "max_output_tokens": 2048,
    "top_p": 0.8,
    "top_k": 40,
}

PROMPT_FORMAT = """ 
    <system_instructions_start1> {system_instruction} </system_instructions_end1>
    <untrusted> {context} </untrusted> 
    <user_input> {question} </user_input> 
    <system_instructions_start2> {system_instruction} </system_instructions_end2>
    """

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
#EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_MODEL_PATH = "./models/qwen3-embedding"
#PROMPT_GUARD_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
PROMPT_GUARD_MODEL_PATH = "./models/deberta-prompt-injection"
DEVICE = "cpu"



