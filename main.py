from knowledge_base import document_chunking
from vector_storage import VectorStore
from llm import Model

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
    if rag is None:
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

            result = rag.generate(query)
            print(f"\nAnswer: {result}")

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()


