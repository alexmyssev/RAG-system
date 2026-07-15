import os
from langchain_text_splitters import RecursiveCharacterTextSplitter#

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=70)
    chunks = splitter.split_text(text)
    return chunks

def document_chunking(folder: str) -> list[dict[str,any]]:
    all_chunks = []

    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), 'r', encoding="utf-8") as f:
                text = f.read()
                chunks = chunk_text(text)

                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        "document": chunk,
                        "metadatas":{
                            "source": filename,
                            "source_path": os.path.join(folder, filename),
                            "chunk_id": i,
                            "status": allowed_access(filename)
                        }
                    })
    return all_chunks

def allowed_access(filename: str) -> str:
    filename_lower = filename.lower()
    if "conf" in filename_lower:
        return "confidential"
    else:
        return "not confidential"

