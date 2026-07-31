from knowledge_base import document_chunking
from vector_storage import VectorStore
from llm import Model
from authorization import Authorization
#from instructions_detector import Detector
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def initialize_RAG():
    doc_folder = "docs"
    vectore_store = VectorStore()

    if vectore_store.collection.count() == 0:
        chunks = document_chunking(doc_folder)

        if not chunks:
            return None
        documents = [c["document"] for c in chunks]
        metadata = [c["metadatas"] for c in chunks]

        vectore_store.add_document(documents, metadata)
        print(f"Added {len(documents)} texts and {len(metadata)} metadata")


    return Model(vectore_store)

def main():
    rag = initialize_RAG()
    #instructions_detector = Detector()

    if rag is None:
        print(f"No documents in the folder")
        return

    role = Authorization().sign_in()

    if role == "admin":
       full_access = True
    else:
        full_access = False


    while True:
        try:

            print(f"Enter 'quit' to stop the program")
            query = input("Enter query: ").strip()
            if query.lower() == "quit":
                break

            if not query:
                continue

            #if instructions_detector.is_malicious(query):
            #    print(f"Malicious query!")
            #    continue

            result = rag.generate(query, full_access)
            print(f"\nAnswer: {result}")

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()


