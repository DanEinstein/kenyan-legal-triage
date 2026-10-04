import os
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from pypdf import PdfReader

PDF_PATH = "./The_Constitution_of_Kenya_2010.pdf"
DB_DIR = "./chroma_kenyan_law_db"
EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_URL = "http://localhost:11434"

def chunk_text(text, chunk_size=1000, overlap=200):
    """Simple python-based chunker."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build_vector_db():
    if not os.path.exists(PDF_PATH):
        print(f"[-] Error: Please place the Constitution PDF at {PDF_PATH}")
        return

    print("[+] Loading Constitution PDF via pypdf...")
    reader = PdfReader(PDF_PATH)
    full_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += f"\n--- Page {i+1} ---\n" + text

    print("[+] Chunking document text...")
    chunks = chunk_text(full_text, chunk_size=1000, overlap=200)

    print("[+] Initializing ChromaDB with Ollama embeddings...")
    client = chromadb.PersistentClient(path=DB_DIR)
    
    try:
        client.delete_collection("kenyan_constitution")
    except Exception:
        pass

    collection = client.create_collection(
        name="kenyan_constitution",
        embedding_function=OllamaEmbeddingFunction(
            url=OLLAMA_URL,
            model_name=EMBEDDING_MODEL,
        ),
    )

    print(f"[+] Embedding and storing {len(chunks)} chunks (running locally via Ollama)...")
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_ids = [f"chunk_{j}" for j in range(i, i+len(batch_chunks))]
        collection.add(
            documents=batch_chunks,
            ids=batch_ids
        )
        print(f" • Processed batch {i // batch_size + 1} / {(len(chunks) + batch_size - 1) // batch_size}")

    print("[✓] Vector database built successfully!")

if __name__ == "__main__":
    build_vector_db()