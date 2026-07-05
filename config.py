AGENT_CONFIG = {
    "temperature": 0.7,
    "max_output_tokens": 2048,
    "top_p": 0.8,
    "top_k": 40,
}

PROMPT_FORMAT = """ 
    Context: {context} 
    Question: {question} 
    """