AGENT_CONFIG = {
    "temperature": 0.7,
    "max_output_tokens": 2048,
    "top_p": 0.8,
    "top_k": 40,
}

PROMPT_FORMAT = """ 
    <system_instructions1> {system_instruction} </system_instructions1>
    <untrusted> {context} </untrusted> 
    <user_input> {question} </user_input> 
    <system_instructions2> {system_instruction} </system_instructions2>
    """

OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_HOST = "http://localhost:11434"
PROMPT_GUARD_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
DEVICE = "cpu"


