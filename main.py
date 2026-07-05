from importlib.metadata import metadata
from generator import Generator
from knowledge_base import document_chunking
from vector_storage import VectorStore

def initialize_RAG():
    doc_folder = "docs"
    vectore_store = VectorStore()

    if vectore_store.collection.count() == 0:
        chunks = document_chunking(doc_folder)

        if not chunks:
            return None
        texts = [c["text"] for c in chunks]
        metadata = [c["metadatas"] for c in chunks]

        vectore_store.add_document(texts, metadata)
        print(f"Added {len(texts)} texts and {len(metadata)} metadata")

        RAG = Generator(vectore_store)
        return RAG

def main():
    RAG = initialize_RAG()
    if RAG is None:
        print(f"No documents in the folder")
        return

    while True:
        try:
            print(f"Enter 'quit' to stop the program")
            query = input("Enter query: ").strip()
            if query.lower() == "quit":
                break

            if not query:
                continue

            result = RAG.generate(query)
            print(f"\nAnswer: {result}")

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()


