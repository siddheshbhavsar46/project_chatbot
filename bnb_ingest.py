import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb import EmbeddingFunction
from uuid import uuid4
from sentence_transformers import SentenceTransformer

load_dotenv()

# =========================
# Config
# =========================
DATA_PATH = "data"
CHROMA_PATH = "chroma"
COLLECTION_NAME = "bitsandbytes_docs"

# =========================
# Embedding function
# =========================
class HFEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def __call__(self, input):
        return self.model.encode(input).tolist()

    def name(self):
        return "hf-mpnet"

# =========================
# Load Markdown files
# =========================
# loader = DirectoryLoader(
#     DATA_PATH,
#     glob="**/*.md",
#     loader_cls=TextLoader
# )

# raw_docs = loader.load()
# print(f"Loaded {len(raw_docs)} markdown files")

# # =========================
# # Chunking
# # =========================
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=600,
#     chunk_overlap=100
# )

# chunks = splitter.split_documents(raw_docs)
# print(f"Created {len(chunks)} chunks")

# # =========================
# # ChromaDB
# # =========================
# client = chromadb.PersistentClient(path=CHROMA_PATH)
# db = client.get_or_create_collection(
#     name=COLLECTION_NAME,
#     embedding_function=HFEmbeddingFunction()
# )

# # =========================
# # Ingest with folder-based metadata
# # =========================
# for chunk in chunks:
#     source = chunk.metadata.get("source", "")

#     # Normalize path
#     normalized_path = source.replace("\\", "/").lower()

#     if "/services/" in normalized_path:
#         doc_type = "service"
#         domain = os.path.splitext(os.path.basename(source))[0]
#     elif "/company/" in normalized_path:
#         doc_type = "company"
#         domain = "company"
#     else:
#         doc_type = "general"
#         domain = "general"

#     db.add(
#         documents=[chunk.page_content],
#         metadatas=[{
#             "source": source,
#             "type": doc_type,
#             "domain": domain,
#             "company": "bitsandbytes"
#         }],
#         ids=[str(uuid4())]
#     )

# print("✅ Markdown ingestion completed successfully!")
# ... (keep your imports and HFEmbeddingFunction class at the top)


def run_ingestion():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    db = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=HFEmbeddingFunction()
    )

    loader = DirectoryLoader(DATA_PATH, glob="**/*.md", loader_cls=TextLoader)
    raw_docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)

    for chunk in chunks:
        # --- THIS SECTION DEFINES THE MISSING VARIABLES ---
        source = chunk.metadata.get("source", "")
        normalized_path = source.replace("\\", "/").lower()

        if "/services/" in normalized_path:
            doc_type = "service"
            domain = os.path.splitext(os.path.basename(source))[0]
        elif "/company/" in normalized_path:
            doc_type = "company"
            domain = "company"
        else:
            doc_type = "general"
            domain = "general"
        # --------------------------------------------------

        db.add(
            documents=[chunk.page_content],
            metadatas=[{
                "source": source,
                "type": doc_type,
                "domain": domain,
                "company": "bitsandbytes"
            }],
            ids=[str(uuid4())]
        )
    
    return f"✅ Success! Ingested {len(chunks)} chunks."

if __name__ == "__main__":
    run_ingestion()