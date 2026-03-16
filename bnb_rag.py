import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from chromadb import EmbeddingFunction
from sentence_transformers import SentenceTransformer

# =========================
# Setup
# =========================
load_dotenv()
genai.configure(api_key="AIzaSyA_S1aPK5eCxwRbeuqNV9BPRjVDFi5pcmw")

model = genai.GenerativeModel('gemini-1.5-flash')

CHROMA_PATH = "chroma"
COLLECTION_NAME = "bitsandbytes_docs"

# =========================
# Embedding function (same as ingestion)
# =========================
class HFEmbedQuery(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def __call__(self, input):
        return self.model.encode(input).tolist()

    def name(self):
        return "hf-mpnet"

# =========================
# Load ChromaDB
# =========================
client = chromadb.PersistentClient(path=CHROMA_PATH)

db = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=HFEmbedQuery()
)

# =========================
# RAG function
# =========================
def rag_answer(question: str) -> str:

    results = db.query(
        query_texts=[question],
        n_results=5,
        where={"company": "bitsandbytes"}  # metadata grounding
    )

    if not results["documents"] or not results["documents"][0]:
        return "I do not have this information at the moment."

    context = "\n\n".join(results["documents"][0])

    prompt = f"""
You are an AI assistant for Bits and Bytes Services.

Answer the user's question strictly using the context below.
Do NOT assume or fabricate information.

If the answer is not present in the context, say:
"I do not have this information at the moment."

Context:
{context}

Question:
{question}
"""

    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(prompt)

    return response.text.strip()

# =========================
# Console loop (testing)
# =========================
if __name__ == "__main__":
    while True:
        q = input("Ask: ").strip()
        if q.lower() in ["exit", "quit"]:
          break
        print("\nAnswer:", rag_answer(q))